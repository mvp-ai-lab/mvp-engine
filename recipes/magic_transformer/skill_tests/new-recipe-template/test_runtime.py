"""Recipe-local runtime tests for the new-recipe-template skill."""

from __future__ import annotations

from pathlib import Path

from mvp_engine.test.recipe_probe import (
    build_engine,
    load_config,
    single_rank_distributed_env,
)
from mvp_engine.utils.log import get_logger
from recipes.magic_transformer.configs.schema import MagicTransformerConfig

RECIPE_PATH = Path("recipes/magic_transformer")
SMALL_MAGIC_TRANSFORMER_OVERRIDE = {
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
}


def build_magic_transformer_engine(*, output_dir: Path, run_id: str, total_steps: int):
    config = load_config(
        RECIPE_PATH,
        output_dir=output_dir,
        run_id=run_id,
        total_steps=total_steps,
        extra_override=SMALL_MAGIC_TRANSFORMER_OVERRIDE,
    )
    return build_engine(RECIPE_PATH, config, config_class=MagicTransformerConfig)


def test_new_recipe_runtime_builds_dataset_model_optimizer_and_scheduler(tmp_path: Path) -> None:
    output_dir = tmp_path / "runtime_outputs"

    with single_rank_distributed_env():
        engine = build_magic_transformer_engine(output_dir=output_dir, run_id="runtime", total_steps=2)
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
