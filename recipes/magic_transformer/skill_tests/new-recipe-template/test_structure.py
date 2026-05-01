"""Recipe-local structure tests for the new-recipe-template skill."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from mvp_engine.engine import ENGINE_REGISTRY
from mvp_engine.utils import skill_testing_util
from recipes.magic_transformer.configs.schema import MagicTransformerConfig
from recipes.magic_transformer.model import MagicTransformer, TransformerConfig

repo_root = skill_testing_util.find_repo_root(Path(__file__))


def recipe_dir() -> Path:
    """Return the recipe directory under test."""
    return repo_root / "recipes" / "magic_transformer"


def import_recipe_modules() -> None:
    """Import recipe modules so the engine registry is populated."""
    import recipes.magic_transformer.configs.schema  # noqa: F401
    import recipes.magic_transformer.dataset  # noqa: F401
    import recipes.magic_transformer.engine  # noqa: F401
    import recipes.magic_transformer.model  # noqa: F401


def test_new_recipe_structure_matches_magic_transformer_scaffold() -> None:
    root = recipe_dir()

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
        "skill_tests/new-recipe-template/test_spec.yaml",
        "skill_tests/new-recipe-template/test_structure.py",
        "skill_tests/new-recipe-template/test_runtime.py",
        "skill_tests/new-recipe-template/test_smoke.py",
    ]

    for relative_path in expected_files:
        assert (root / relative_path).exists(), relative_path

    readme_text = (root / "README.md").read_text(encoding="utf-8")
    assert "# Magic Transformer" in readme_text
    assert "fake autoregressive token dataset" in readme_text
    assert "TODO" not in readme_text

    raw_config = OmegaConf.load(root / "configs" / "train.yaml")
    config = MagicTransformerConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))
    assert config.project.name == "magic_transformer"
    assert config.engine == "MagicTransformerEngine"
    assert config.checkpoint.interval >= 1
    assert config.log.backends

    import_recipe_modules()
    engine_cls = ENGINE_REGISTRY.get("MagicTransformerEngine")
    assert engine_cls.__name__ == "MagicTransformerEngine"
    assert hasattr(engine_cls, "prepare_logger")
    assert hasattr(engine_cls, "save")
    assert hasattr(engine_cls, "load")

    assert MagicTransformer.__name__ == "MagicTransformer"
    assert TransformerConfig.__name__ == "TransformerConfig"
