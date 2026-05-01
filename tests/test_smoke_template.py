"""Template for recipe-local skill smoke tests.

Copy this file into ``recipes/<recipe>/skill_tests/<skill-id>/test_smoke.py``
and update the import block first. The default imports target the
``magic_transformer`` recipe.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import torch.distributed as dist
from omegaconf import OmegaConf

from mvp_engine.engine import ENGINE_REGISTRY
from mvp_engine.utils import skill_testing_util
from mvp_engine.utils.log import get_logger

# Update this import block when copying the template to a new recipe.
from recipes.magic_transformer.configs.schema import MagicTransformerConfig

repo_root = skill_testing_util.find_repo_root(Path(__file__))

if Path(__file__).name.endswith("_template.py"):
    pytestmark = pytest.mark.skip(reason="Template file. Copy and rename into a recipe-local skill_tests directory.")


def recipe_dir() -> Path:
    """Return the default recipe directory used by the template."""
    return repo_root / "recipes" / "magic_transformer"


def import_recipe_modules() -> None:
    """Import recipe modules so the engine registry is populated."""
    import recipes.magic_transformer.configs.schema  # noqa: F401
    import recipes.magic_transformer.dataset  # noqa: F401
    import recipes.magic_transformer.engine  # noqa: F401
    import recipes.magic_transformer.model  # noqa: F401


def load_recipe_config(
    *,
    output_dir: Path,
    run_id: str,
    total_steps: int,
    config_name: str = "train",
    extra_override: dict | None = None,
):
    """Load the default recipe config with runtime-safe test overrides."""
    config_path = recipe_dir() / "configs" / f"{config_name}.yaml"
    base_config = OmegaConf.load(config_path)
    override = OmegaConf.create(
        {
            "runtime": {"run_id": run_id},
            "project": {"dir": str(output_dir)},
            "log": {"interval": 1, "backends": ["file"]},
            "loop": {"total_steps": total_steps},
            "checkpoint": {"interval": 1, "keep_n": 1},
            "optim": {"mixed_precision": "fp32"},
            "data": {
                "fake_train_size": 8,
                "fake_eval_size": 4,
                "batch_size": 2,
                "num_workers": 0,
                "seq_len": 8,
                "vocab_size": 128,
            },
            "model": {
                "vocab_size": 128,
                "max_seq_len": 8,
                "d_model": 32,
                "n_heads": 4,
                "n_kv_heads": 2,
                "n_layers": 2,
                "dropout": 0.0,
            },
            "parallel": {"mesh": {"replicate": 1, "shard": 1, "tensor": 1}},
        }
    )
    if extra_override is None:
        return OmegaConf.merge(base_config, override)
    return OmegaConf.merge(base_config, override, OmegaConf.create(extra_override))


def build_engine(
    *,
    output_dir: Path,
    run_id: str = "skill-smoke",
    total_steps: int = 1,
    config_name: str = "train",
    extra_override: dict | None = None,
):
    """Construct the default recipe engine with a small deterministic config."""
    import_recipe_modules()
    config = load_recipe_config(
        output_dir=output_dir,
        run_id=run_id,
        total_steps=total_steps,
        config_name=config_name,
        extra_override=extra_override,
    )
    config_model = MagicTransformerConfig.model_validate(OmegaConf.to_container(config, resolve=True))
    engine_cls = ENGINE_REGISTRY.get(config_model.engine)
    return engine_cls(config)


def build_skill_test_command(
    *,
    skill_id: str,
    layer: str | None = None,
    recipe_name: str = "magic_transformer",
) -> str:
    """Return the canonical skill test command for the selected recipe and skill."""
    return skill_testing_util.get_default_skill_test_command(
        recipe_name,
        skill_id=skill_id,
        layer=layer,
    )


def require_cuda_gpus_for_skill(
    *,
    skill_id: str,
    layer: str,
    count: int,
    recipe_name: str = "magic_transformer",
) -> None:
    """Raise a real-env requirement when the requested CUDA GPU count is unavailable."""
    command = build_skill_test_command(skill_id=skill_id, layer=layer, recipe_name=recipe_name)
    try:
        import torch
    except Exception:
        skill_testing_util.raise_real_env_required(
            command=command,
            reason=f"{skill_id} {layer} requires {count} CUDA GPU(s), but PyTorch CUDA is unavailable.",
        )
        return

    if not torch.cuda.is_available():
        skill_testing_util.raise_real_env_required(
            command=command,
            reason=f"{skill_id} {layer} requires visible CUDA devices.",
        )
    if torch.cuda.device_count() < count:
        skill_testing_util.raise_real_env_required(
            command=command,
            reason=f"{skill_id} {layer} requires {count} CUDA GPU(s), found {torch.cuda.device_count()}.",
        )


@contextmanager
def single_rank_distributed_env() -> Iterator[None]:
    """Provide minimal env:// settings for one-rank engine initialization."""
    previous_env = {
        key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = sock.getsockname()[1]

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def multi_rank_distributed_env(
    *,
    rank: int,
    world_size: int,
    local_rank: int | None = None,
    master_addr: str = "127.0.0.1",
    master_port: int,
) -> Iterator[None]:
    """Provide env:// settings for one worker in a multi-rank smoke test."""
    previous_env = {
        key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    }

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank if local_rank is None else local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_recipe_smoke_template_runs_one_real_training_step_and_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke_outputs"

    with single_rank_distributed_env():
        engine = build_engine(output_dir=output_dir, run_id="smoke", total_steps=1)
        try:
            engine.before_train()

            batch = next(iter(engine.train_loader))
            outputs = engine.train_one_step(engine.train_pre_step(batch))

            assert outputs["loss"].requires_grad
            assert outputs["loss"].item() > 0

            engine.train_after_step(outputs)
            assert engine.step == 1

            engine.after_train()

            run_dir = Path(engine.project_dir)
            log_file = run_dir / f"log_{engine.run_id}.log"
            checkpoint_dir = run_dir / "checkpoints" / "iter_1"

            assert log_file.exists()
            assert checkpoint_dir.exists()
            assert (checkpoint_dir / "engine.pt").exists()
        finally:
            logger_instance = get_logger()
            if logger_instance is not None:
                logger_instance.destroy()


# For distributed skills, replace ``single_rank_distributed_env()`` with
# ``multi_rank_distributed_env(...)`` inside a real multi-process launcher or
# ``torch.multiprocessing`` worker. Configure the copied smoke test to match the
# required distributed mode such as DDP, FSDP2 shard, or tensor parallel.
