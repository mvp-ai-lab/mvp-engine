"""Recipe-local smoke tests for the new-recipe-template skill."""

from __future__ import annotations

from pathlib import Path

from mvp_engine.engine import TrainStepContext
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


def test_new_recipe_smoke_runs_one_real_training_step_and_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke_outputs"

    with single_rank_distributed_env():
        engine = build_magic_transformer_engine(output_dir=output_dir, run_id="smoke", total_steps=1)
        try:
            engine.before_train()

            ctx = TrainStepContext(
                data=next(iter(engine.train_loader)),
                step=engine.step,
                epoch=engine.epoch,
                micro_step=engine.ga_state.micro_step,
            )
            prepared = engine.train_pre_step(ctx)
            if prepared is not None and prepared is not ctx:
                ctx.data = prepared

            engine.train_exec_step(ctx)

            assert ctx.loss is not None
            assert ctx.loss.requires_grad
            assert ctx.loss.item() > 0

            engine.train_post_step(ctx)
            assert engine.step == 1

            engine.after_train()

            run_dir = Path(engine.project_dir)
            log_file = run_dir / f"log_{engine.run_id}.log"
            checkpoint_dir = run_dir / "checkpoints" / "iter_1"

            assert log_file.exists()
            assert "train/loss" in log_file.read_text(encoding="utf-8")
            assert checkpoint_dir.exists()
            assert (checkpoint_dir / "model.pt").exists()
            assert (checkpoint_dir / "optimizer.pt").exists()
            assert (checkpoint_dir / "engine.pt").exists()
        finally:
            logger_instance = get_logger()
            if logger_instance is not None:
                logger_instance.destroy()
