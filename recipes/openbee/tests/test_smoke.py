"""Smoke tests for the OpenBee recipe.

The smoke test runs the real ``engine.train()`` path in spawned distributed
worker processes. Recipe-local skill assertions are injected dynamically around
engine phases, so skills can validate runtime behavior without adding hook code
to the production engine.
"""

import traceback
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

from mvp_engine.testing.utils import (
    distributed_env,
    find_free_port,
    inject_engine_assert_hooks,
    load_recipe_skill_asserts,
    load_smoke_test_config,
)


def smoke_process(
    rank: int,
    world_size: int,
    master_port: int,
    config_name: str,
    output_dir: Path,
    config_overrides: tuple[str, ...],
    result_queue,
) -> None:
    """Run one distributed smoke worker and report its traceback to the parent.

    Each worker imports recipe modules, loads the selected config with standard
    smoke-test overrides, injects skill assertion hooks into the engine
    instance, and executes ``engine.train()``. Any exception is sent through
    ``result_queue`` before being re-raised so pytest reports the real worker
    traceback instead of only a process exit code.
    """
    try:
        with distributed_env(
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            master_port=master_port,
            master_addr="127.0.0.1",
        ):
            from mvp_engine.engine import ENGINE_REGISTRY
            from mvp_engine.launch import _import_recipe_modules

            _import_recipe_modules(Path(__file__).resolve().parent.parent)

            recipe_root = Path(__file__).resolve().parent.parent
            config = load_smoke_test_config(
                recipe_root,
                config_name,
                output_dir=output_dir,
                config_overrides=config_overrides,
            )

            engine = ENGINE_REGISTRY.get(config.engine)(config)
            inject_engine_assert_hooks(engine, load_recipe_skill_asserts(recipe_root))
            engine.train()

        result_queue.put((rank, True, ""))
    except BaseException:
        result_queue.put((rank, False, traceback.format_exc()))
        raise


def test_smoke(
    world_size: int,
    config_name: str,
    config_overrides: tuple[str, ...],
    tmp_path: Path,
):
    """Run OpenBee smoke validation across ``world_size`` spawned workers.

    The test uses ``tmp_path`` as an isolated output root and passes
    ``config_overrides`` through to the smoke config loader. Multi-process
    failures are collected from both worker exit codes and traceback messages
    returned over the result queue.
    """
    master_port = find_free_port()
    output_dir = tmp_path / "smoke_outputs"

    context = get_context("spawn")
    result_queue = context.Queue()

    processes = [
        context.Process(
            target=smoke_process,
            args=(
                rank,
                world_size,
                master_port,
                config_name,
                output_dir,
                config_overrides,
                result_queue,
            ),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    for process in processes:
        process.join(12000)

    errors = []
    for process in processes:
        if process.is_alive():
            process.terminate()
            errors.append(f"rank process {process.pid} timed out.")
        elif process.exitcode != 0:
            errors.append(f"rank process {process.pid} exited with code {process.exitcode}.")

    while True:
        try:
            rank, passed, message = result_queue.get_nowait()
        except Empty:
            break
        if not passed:
            errors.append(f"rank {rank} failed:\n{message}")

    assert not errors, "\n".join(errors)
