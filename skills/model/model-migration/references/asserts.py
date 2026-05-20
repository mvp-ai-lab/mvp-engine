"""Recipe-local assertions for the model-migration skill.

Copy this file to:
recipes/<recipe>/tests/skills/model-migration/asserts.py
"""

import importlib
import inspect
import textwrap
from pathlib import Path

from mvp_engine.testing.utils import read_recipe_source


def test_file_structure(recipe_root: Path) -> None:
    """Verify migrated model files and recipe-local validation hooks exist."""
    model_dir = recipe_root / "model"
    assert model_dir.exists(), "Migrated model code must live under recipes/<recipe>/model/."
    assert (recipe_root / "tests/skills/model-migration/asserts.py").exists(), (
        "Copy model-migration references/asserts.py into the recipe-local tests/skills path."
    )

    model_files = [path for path in model_dir.rglob("*.py") if "__pycache__" not in path.parts]
    assert model_files, "recipes/<recipe>/model/ must contain Python model files."

    source = read_recipe_source(recipe_root)
    assert "state_dict" in source or "from_pretrained" in source or "load_state_dict" in source, (
        "Migration must expose checkpoint loading or state_dict compatibility logic."
    )
    assert "__init__.py" in {path.name for path in model_files}, "recipes/<recipe>/model/__init__.py must exist."


def test_config_structure(config) -> None:
    """Verify recipe config exposes a model section for migrated model construction."""
    assert hasattr(config, "model"), "Recipe config must expose a model section."
    model_config = config.model
    field_names = set(model_config.keys()) if hasattr(model_config, "keys") else set(vars(model_config))
    identity_fields = ("pretrained_model_name_or_path", "model_name_or_path", "checkpoint_path", "model_type")
    has_model_identity = any(field in field_names for field in identity_fields)
    assert has_model_identity, (
        "model config should expose a checkpoint/model identity field; adapt this assertion for custom configs."
    )


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine constructs a recipe-local model through prepare_model."""
    assert hasattr(engine_class, "prepare_model"), "Engine must implement prepare_model."
    module_name = engine_class.__module__.rsplit(".", 1)[0].rsplit(".", 1)[0]
    assert module_name.startswith("recipes."), "Engine must be recipe-local."
    prepare_model_source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    assert "model" in prepare_model_source.lower(), "prepare_model must construct or return the migrated model."


def assert_before_train_start(engine) -> None:
    """Before training, verify recipe model package imports cleanly."""
    recipe_module = engine.__class__.__module__.split(".engine.", 1)[0]
    importlib.import_module(f"{recipe_module}.model")


def assert_before_train_end(engine) -> None:
    """After setup, verify the runtime model has parameters and a stable state dict."""
    model = engine.model.module if hasattr(engine.model, "module") else engine.model
    state_dict = model.state_dict()
    assert state_dict, "Migrated runtime model state_dict is empty."
    assert list(model.parameters()), "Migrated runtime model has no parameters."
