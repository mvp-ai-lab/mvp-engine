"""Assertions contributed by the new-recipe-template skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mvp_engine.utils.log import get_logger
from recipes.magic_transformer.model import MagicTransformer, TransformerConfig


def assert_structure(*, recipe_root: Path, config: Any, engine_cls: type) -> None:
    expected_files = [
        "README.md",
        "__init__.py",
        "configs/__init__.py",
        "configs/schema.py",
        "configs/train.yaml",
        "dataset/__init__.py",
        "dataset/dataset.py",
        "dataset/sampler.py",
        "engine/__init__.py",
        "engine/magic_transformer_engine.py",
        "model/__init__.py",
        "model/builder.py",
        "model/magic_transformer.py",
        "model/source_model.py",
        "skill_tests/skill_manifest.yaml",
        "skill_tests/test_structure.py",
        "skill_tests/test_smoke.py",
        "skill_tests/new-recipe-template/asserts.py",
    ]

    for relative_path in expected_files:
        assert (recipe_root / relative_path).exists(), relative_path

    readme_text = (recipe_root / "README.md").read_text(encoding="utf-8")
    assert "# Magic Transformer" in readme_text
    assert "fake autoregressive token dataset" in readme_text
    assert "TODO" not in readme_text

    assert config.project.name == "magic_transformer"
    assert config.engine == "MagicTransformerEngine"
    assert config.checkpoint.interval >= 1
    assert config.log.backends

    assert engine_cls.__name__ == "MagicTransformerEngine"
    assert hasattr(engine_cls, "prepare_logger")
    assert hasattr(engine_cls, "save")
    assert hasattr(engine_cls, "load")

    assert MagicTransformer.__name__ == "MagicTransformer"
    assert TransformerConfig.__name__ == "TransformerConfig"


def assert_smoke(
    *,
    engine: Any,
    ctx: Any,
    batch: dict[str, Any],
    log_file: Path,
    checkpoint_dir: Path,
) -> None:
    unwrapped_model = engine.unwrapped_model

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].ndim == 2
    assert unwrapped_model.__class__.__name__ == "MagicTransformer"
    assert engine.optimizer.__class__.__name__ == "AdamW"
    assert engine.scheduler.__class__.__name__ in {"SequentialLR", "CosineAnnealingLR"}
    assert get_logger() is not None
    assert ctx.loss is not None
    assert ctx.loss.requires_grad
    assert ctx.loss.item() > 0
    assert engine.step == 1

    assert log_file.exists()
    assert "train/loss" in log_file.read_text(encoding="utf-8")
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "model.pt").exists()
    assert (checkpoint_dir / "optimizer.pt").exists()
    assert (checkpoint_dir / "engine.pt").exists()
