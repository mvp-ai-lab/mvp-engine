"""Recipe-level cumulative smoke tests for installed skills."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

from mvp_engine.engine import TrainStepContext
from mvp_engine.test.recipe_probe import (
    build_engine,
    load_config,
    single_rank_distributed_env,
)
from mvp_engine.utils import skill_testing_util
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


def test_recipe_smoke_matches_installed_skill_assertions(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke_outputs"

    with single_rank_distributed_env():
        engine = _build_magic_transformer_engine(output_dir=output_dir, run_id="smoke", total_steps=1)
        try:
            engine.before_train()
            batch = next(iter(engine.train_loader))
            skill_asserts = _load_skill_asserts()

            ctx = TrainStepContext(
                data=batch,
                step=engine.step,
                epoch=engine.epoch,
                micro_step=engine.ga_state.micro_step,
            )
            prepared = engine.train_pre_step(ctx)
            if prepared is not None and prepared is not ctx:
                ctx.data = prepared

            engine.train_exec_step(ctx)
            engine.train_post_step(ctx)
            engine.after_train()

            run_dir = Path(engine.project_dir)
            log_file = run_dir / f"log_{engine.run_id}.log"
            checkpoint_dir = run_dir / "checkpoints" / "iter_1"
            for skill_id, asserts in skill_asserts:
                assert_smoke = asserts.get("assert_smoke")
                if assert_smoke is None:
                    raise AssertionError(f"{skill_id} asserts.py must define assert_smoke.")
                assert_smoke(engine=engine, ctx=ctx, batch=batch, log_file=log_file, checkpoint_dir=checkpoint_dir)
        finally:
            logger_instance = get_logger()
            if logger_instance is not None:
                logger_instance.destroy()


def _build_magic_transformer_engine(*, output_dir: Path, run_id: str, total_steps: int):
    config = load_config(
        RECIPE_PATH,
        output_dir=output_dir,
        run_id=run_id,
        total_steps=total_steps,
        extra_override=SMALL_MAGIC_TRANSFORMER_OVERRIDE,
    )
    return build_engine(RECIPE_PATH, config, config_class=MagicTransformerConfig)


def _load_skill_asserts() -> tuple[tuple[str, dict[str, Any]], ...]:
    return tuple(
        (skill_id, runpy.run_path(str(asserts_path)))
        for skill_id, asserts_path in skill_testing_util.get_ordered_skill_asserts(RECIPE_PATH)
    )
