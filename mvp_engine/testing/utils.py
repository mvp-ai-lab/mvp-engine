"""Shared test utilities for recipe-local pytest suites.

This module is intentionally limited to test infrastructure. It provides helpers
for distributed smoke tests, recipe-local skill assertion loading, temporary
smoke-test config overrides, and dynamic engine method hooks used by tests.
"""

import importlib.util
import inspect
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from mvp_engine.utils.misc import get_device

DIST_ENV_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
ENGINE_HOOK_PHASES = (
    "before_train",
    "do_train",
    "train_pre_step",
    "train_exec_step",
    "forward_step",
    "backward_step",
    "optimizer_step",
    "train_post_step",
    "after_train",
)


@contextmanager
def distributed_env(
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    master_port: int,
    master_addr: str = "127.0.0.1",
):
    """Temporarily set distributed env vars for one test worker process.

    The context preserves the caller's original distributed environment and
    restores it on exit. It also destroys a process group initialized inside the
    context, so each smoke worker can leave the process cleanly.

    This helper expects a real accelerator-backed distributed environment. CPU
    fallback is rejected because these recipe smoke tests are intended to
    validate the same distributed path used by training.
    """
    previous_env = {key: os.environ.get(key) for key in DIST_ENV_KEYS}
    was_initialized = dist.is_initialized()

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    if get_device(local_rank).type == "cpu" and not dist.is_initialized():
        raise RuntimeError("Distributed tests require GPUs/NPUs.")

    try:
        yield
    finally:
        if not was_initialized and dist.is_initialized():
            dist.destroy_process_group()
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def find_free_port():
    """Return a free localhost TCP port for distributed test rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def load_smoke_test_config(
    recipe_root: Path,
    config_name: str,
    *,
    output_dir: Path,
    run_id: str = "smoke",
    config_overrides: tuple[str, ...] = (),
    extra_override: dict | DictConfig | None = None,
) -> DictConfig:
    """Load a recipe config with standard smoke-test overrides.

    The base config is loaded from ``recipes/<recipe>/configs/<config_name>.yaml``.
    A small smoke-test override is then merged in to keep outputs isolated,
    reduce the run to one training step, disable heavyweight checkpoint export,
    and force dataloading to stay in the current process.

    The helper deliberately does not override ``parallel.mesh``. DP/FSDP/TP
    layout is part of the behavior under test and should be supplied by the
    recipe config or by explicit ``config_overrides`` such as
    ``parallel.mesh.tensor=2``.

    Merge order is base config, standard smoke override, optional structured
    ``extra_override``, then command-line dotlist ``config_overrides``.
    Later values win.
    """
    config_path = recipe_root / "configs" / f"{config_name}.yaml"
    assert config_path.exists(), f"Config file does not exist: {config_path}"

    override = {
        "project": {"dir": str(output_dir)},
        "runtime": {"run_id": run_id},
        "log": {"interval": 1, "backends": ["terminal", "file"]},
        "checkpoint": {"interval": 1, "keep_n": 1, "hf_enable": False},
        "data": {"num_workers": 0},
        "loop": {"total_steps": 1},
    }
    config = OmegaConf.load(config_path)
    overrides = [config, OmegaConf.create(override)]
    if extra_override is not None:
        overrides.append(OmegaConf.create(extra_override))
    if config_overrides:
        overrides.append(OmegaConf.from_dotlist(list(config_overrides)))
    return OmegaConf.merge(*overrides)


def inject_engine_hook(engine, method_name: str, *, before=None, after=None):
    """Wrap one engine instance method with optional before and after callbacks.

    The wrapper is installed only on the provided engine instance. It does not
    modify the engine class or production code. ``before`` receives
    ``(engine, *args, **kwargs)`` and ``after`` receives
    ``(engine, result, *args, **kwargs)``.
    """
    original = getattr(engine, method_name)

    def wrapped(*args, **kwargs):
        if before is not None:
            before(engine, *args, **kwargs)

        result = original(*args, **kwargs)

        if after is not None:
            after(engine, result, *args, **kwargs)

        return result

    setattr(engine, method_name, wrapped)


def load_recipe_skill_asserts(recipe_root: Path) -> list[ModuleType]:
    """Load recipe-local skill assertion modules for a recipe.

    Assertion modules are discovered under
    ``<recipe_root>/tests/skills/<skill-id>/asserts.py`` and loaded as isolated
    Python modules. The returned modules are consumed by structure and smoke
    tests to extend the recipe's baseline assertions with skill-specific checks.
    """
    recipe_name = recipe_root.name
    skill_asserts = []
    for asserts_path in sorted((recipe_root / "tests" / "skills").glob("*/asserts.py")):
        module_name = f"{recipe_name}_{asserts_path.parent.name.replace('-', '_')}_asserts"
        spec = importlib.util.spec_from_file_location(module_name, asserts_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        skill_asserts.append(module)
    return skill_asserts


def inject_engine_assert_hooks(engine: object, skill_asserts: list[ModuleType]) -> None:
    """Inject skill assertion callbacks around standard engine phases.

    For each phase in ``ENGINE_HOOK_PHASES``, this helper looks for assertion
    functions with the following names in each skill module:

    - ``assert_<phase>_start``
    - ``assert_<phase>_end``

    For example, ``assert_forward_step_end(engine, ctx)`` runs after
    ``engine.forward_step(ctx)``. The payload can include ``engine``, ``ctx``,
    ``result``, ``args``, and ``kwargs``; assertion functions only need to
    declare the parameters they use.
    """
    for phase in ENGINE_HOOK_PHASES:
        start_name = f"assert_{phase}_start"
        end_name = f"assert_{phase}_end"
        if not any(
            hasattr(asserts_module, start_name) or hasattr(asserts_module, end_name) for asserts_module in skill_asserts
        ):
            continue

        def before(engine, *args, _assert_name=start_name, **kwargs):
            payload = {"engine": engine, "args": args, "kwargs": kwargs}
            if args:
                payload["ctx"] = args[0]
            _run_skill_asserts(skill_asserts, _assert_name, **payload)

        def after(engine, result, *args, _assert_name=end_name, **kwargs):
            payload = {"engine": engine, "args": args, "kwargs": kwargs, "result": result}
            if args:
                payload["ctx"] = args[0]
            _run_skill_asserts(skill_asserts, _assert_name, **payload)

        inject_engine_hook(engine, phase, before=before, after=after)


def _run_skill_asserts(skill_asserts: list[ModuleType], assert_name: str, **payload) -> None:
    """Run matching assertion functions with only the parameters they accept."""
    for asserts_module in skill_asserts:
        assert_func = getattr(asserts_module, assert_name, None)
        if assert_func is None:
            continue
        signature = inspect.signature(assert_func)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            assert_func(**payload)
        else:
            assert_func(**{name: payload[name] for name in signature.parameters if name in payload})
