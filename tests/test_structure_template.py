"""Template for recipe-local skill structure tests.

Copy this file into ``recipes/<recipe>/skill_tests/<skill-id>/test_structure.py``
and update the import block first. The default imports target the
``magic_transformer`` recipe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Update this import block when copying the template to a new recipe.
from magic_transformer.configs.schema import MagicTransformerConfig
from magic_transformer.model import MagicTransformer, TransformerConfig
from omegaconf import OmegaConf

from mvp_engine.engine import ENGINE_REGISTRY

repo_root = Path(__file__).resolve().parents[1]

if Path(__file__).name.endswith("_template.py"):
    pytestmark = pytest.mark.skip(reason="Template file. Copy and rename into a recipe-local skill_tests directory.")


def recipe_dir() -> Path:
    """Return the default recipe directory used by the template."""
    return repo_root / "recipes" / "magic_transformer"


def import_recipe_modules() -> None:
    """Import recipe modules so the engine registry is populated."""
    import magic_transformer.configs.schema  # noqa: F401
    import magic_transformer.dataset  # noqa: F401
    import magic_transformer.engine  # noqa: F401
    import magic_transformer.model  # noqa: F401


def test_recipe_structure_template_matches_skill_test_scaffold() -> None:
    root = recipe_dir()
    skill_dir_name = Path(__file__).resolve().parent.name

    expected_files = [
        "README.md",
        "__init__.py",
        "configs/__init__.py",
        "configs/schema.py",
        "configs/train.yaml",
        "dataset/__init__.py",
        "engine/__init__.py",
        "model/__init__.py",
        "skill_tests/skill_manifest.yaml",
        f"skill_tests/{skill_dir_name}/test_spec.yaml",
        f"skill_tests/{skill_dir_name}/test_structure.py",
        f"skill_tests/{skill_dir_name}/test_runtime.py",
        f"skill_tests/{skill_dir_name}/test_smoke.py",
    ]

    for relative_path in expected_files:
        assert (root / relative_path).exists(), relative_path

    raw_config = OmegaConf.load(root / "configs" / "train.yaml")
    config = MagicTransformerConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))

    assert config.project.name
    assert config.engine
    assert config.checkpoint.interval >= 1
    assert config.log.backends

    import_recipe_modules()
    engine_cls = ENGINE_REGISTRY.get(config.engine)
    assert engine_cls.__name__ == config.engine
    assert hasattr(engine_cls, "prepare_logger")
    assert hasattr(engine_cls, "save")
    assert hasattr(engine_cls, "load")

    assert MagicTransformer.__name__
    assert TransformerConfig.__name__
