"""Recipe-local runtime tests for the new-recipe-template skill."""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch.distributed as dist
from omegaconf import OmegaConf

from mvp_engine.engine import ENGINE_REGISTRY
from mvp_engine.utils import skill_testing_util
from mvp_engine.utils.log import get_logger
from recipes.magic_transformer.configs.schema import MagicTransformerConfig

repo_root = Path(__file__).resolve().parents[4]


def recipe_dir() -> Path:
    """Return the recipe directory under test."""
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
    run_id: str = "skill-runtime",
    total_steps: int = 2,
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
    language: str = "en",
) -> str:
    """Return the canonical skill test command for this recipe-local skill."""
    return skill_testing_util.get_default_skill_test_command(
        recipe_name,
        language=language,
        skill_id=skill_id,
        layer=layer,
    )


@contextmanager
def single_rank_distributed_env(*, skill_id: str, layer: str) -> Iterator[None]:
    """Provide minimal env:// settings for one-rank engine initialization."""
    previous_env = {
        key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    }
    command = build_skill_test_command(skill_id=skill_id, layer=layer)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            master_port = sock.getsockname()[1]
    except PermissionError as exc:
        skill_testing_util.raise_real_env_required(
            command=command,
            reason=f"{skill_id} {layer} needs local socket bind permission: {exc}",
        )

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


def test_new_recipe_runtime_builds_dataset_model_optimizer_and_scheduler(tmp_path: Path) -> None:
    output_dir = tmp_path / "runtime_outputs"

    with single_rank_distributed_env(skill_id="new-recipe-template", layer="runtime"):
        engine = build_engine(output_dir=output_dir, run_id="runtime", total_steps=2)
        try:
            engine.before_train()

            batch = next(iter(engine.train_loader))
            unwrapped_model = engine.unwrapped_model

            assert batch["input_ids"].shape == batch["labels"].shape
            assert batch["input_ids"].ndim == 2
            assert unwrapped_model.__class__.__name__ == "MagicTransformer"
            assert engine.optimizer.__class__.__name__ == "AdamW"
            assert engine.scheduler.__class__.__name__ in {"SequentialLR", "CosineAnnealingLR"}
            assert get_logger() is not None
        finally:
            logger_instance = get_logger()
            if logger_instance is not None:
                logger_instance.destroy()
