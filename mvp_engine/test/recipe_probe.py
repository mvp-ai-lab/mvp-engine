"""Shared helpers for recipe-local engine tests."""

from __future__ import annotations

import importlib
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from mvp_engine.engine import ENGINE_REGISTRY
from mvp_engine.utils import skill_testing_util
from mvp_engine.utils.misc import get_device

DEFAULT_RECIPE_MODULES = ("configs.schema", "dataset", "engine", "model")
DIST_ENV_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


def import_modules(
    recipe_path: str | Path,
    module_names: tuple[str, ...] = DEFAULT_RECIPE_MODULES,
) -> dict[str, ModuleType]:
    """Import recipe modules relative to a recipe directory or recipe name."""
    recipe_package = _recipe_package_name(recipe_path)
    return {module_name: importlib.import_module(f"{recipe_package}.{module_name}") for module_name in module_names}


def load_config(
    recipe_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    total_steps: int | None = None,
    config_name: str = "train",
    extra_override: dict | DictConfig | None = None,
) -> DictConfig:
    """Load a recipe config with common test-time overrides."""
    recipe_dir = _resolve_recipe_dir(recipe_path)
    config = OmegaConf.load(recipe_dir / "configs" / f"{config_name}.yaml")
    override = _build_test_override(output_dir=output_dir, run_id=run_id, total_steps=total_steps)

    if extra_override is None:
        return OmegaConf.merge(config, OmegaConf.create(override))
    return OmegaConf.merge(config, OmegaConf.create(override), OmegaConf.create(extra_override))


def build_engine(
    recipe_path: str | Path,
    config: DictConfig,
    *,
    config_class: type | None = None,
):
    """Import recipe modules, resolve the configured engine class, and instantiate it."""
    import_modules(recipe_path)
    config_payload = OmegaConf.to_container(config, resolve=True)
    config_model = config_class.model_validate(config_payload) if config_class is not None else config
    engine_name = config_model.engine
    engine_cls = ENGINE_REGISTRY.get(engine_name)
    return engine_cls(config)


@contextmanager
def single_rank_distributed_env() -> Iterator[None]:
    """Provide env:// settings for one-rank engine initialization."""
    with _distributed_env(rank=0, world_size=1, local_rank=0, master_port=_find_free_port()):
        yield


@contextmanager
def multi_rank_distributed_env(
    *,
    rank: int,
    world_size: int,
    master_port: int,
    local_rank: int | None = None,
    master_addr: str = "127.0.0.1",
) -> Iterator[None]:
    """Provide env:// settings for one worker in a multi-rank test."""
    with _distributed_env(
        rank=rank,
        world_size=world_size,
        local_rank=rank if local_rank is None else local_rank,
        master_addr=master_addr,
        master_port=master_port,
    ):
        yield


def _build_test_override(
    *,
    output_dir: str | Path | None,
    run_id: str | None,
    total_steps: int | None,
) -> dict:
    override: dict = {
        "log": {"interval": 1, "backends": ["file"]},
        "checkpoint": {"interval": 1, "keep_n": 1},
        "optim": {"mixed_precision": "fp32"},
        "parallel": {"mesh": {"replicate": 1, "shard": 1, "tensor": 1}},
    }
    if output_dir is not None:
        override["project"] = {"dir": str(output_dir)}
    if run_id is not None:
        override["runtime"] = {"run_id": run_id}
    if total_steps is not None:
        override["loop"] = {"total_steps": total_steps}
    return override


@contextmanager
def _distributed_env(
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    master_port: int,
    master_addr: str = "127.0.0.1",
) -> Iterator[None]:
    previous_env = {key: os.environ.get(key) for key in DIST_ENV_KEYS}
    was_initialized = dist.is_initialized()

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    if get_device(local_rank).type == "cpu" and not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://", world_size=world_size, rank=rank)

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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _recipe_package_name(recipe_path: str | Path) -> str:
    recipe_dir = _resolve_recipe_dir(recipe_path)
    relative_path = recipe_dir.relative_to(skill_testing_util.find_repo_root(recipe_dir))
    return ".".join(relative_path.parts)


def _resolve_recipe_dir(recipe_path: str | Path) -> Path:
    repo_root = skill_testing_util.find_repo_root()
    path = Path(recipe_path)
    if path.is_absolute():
        candidate = path
    elif path.parts[:1] == ("recipes",):
        candidate = repo_root / path
    else:
        candidate = repo_root / "recipes" / path

    if not candidate.exists():
        raise skill_testing_util.SkillTestSpecError(f"Recipe path does not exist: {candidate}")
    if not candidate.is_dir():
        raise skill_testing_util.SkillTestSpecError(f"Recipe path is not a directory: {candidate}")
    return candidate.resolve()
