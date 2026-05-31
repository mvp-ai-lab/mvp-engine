"""Structure tests for the video MLLM recipe."""

import importlib
from pathlib import Path

from omegaconf import OmegaConf

from mvp_engine.testing.utils import load_recipe_skill_asserts

RECIPE_IMPORT_PATH = "recipes.video_mllm"
CONFIG_SCHEMA_MODULE = f"{RECIPE_IMPORT_PATH}.configs.schema"
CONFIG_CLASS_NAME = "VideoMLLMConfig"

EXPECTED_FILES = [
    "README.md",
    "__init__.py",
    "configs/__init__.py",
    "configs/schema.py",
    "configs/train.yaml",
    "dataset/__init__.py",
    "dataset/decoder.py",
    "dataset/sampling.py",
    "dataset/preprocess.py",
    "dataset/processor.py",
    "dataset/collator.py",
    "dataset/dataset.py",
    "dataset/types.py",
    "engine/__init__.py",
    "engine/video_mllm_engine.py",
    "guards/__init__.py",
    "guards/loss.py",
    "model/__init__.py",
    "model/qwen3_vl.py",
]


def _recipe_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_config_class():
    module = importlib.import_module(CONFIG_SCHEMA_MODULE)
    return getattr(module, CONFIG_CLASS_NAME)


def _iter_config_paths(recipe_root: Path):
    return sorted((recipe_root / "configs").glob("*.yaml"))


def test_file_structure():
    """Validate the baseline recipe file layout."""
    recipe_root = _recipe_root()

    for relative_path in EXPECTED_FILES:
        assert (recipe_root / relative_path).exists(), f"{relative_path} does not exist in the recipe root."

    for asserts_module in load_recipe_skill_asserts(recipe_root):
        if hasattr(asserts_module, "test_file_structure"):
            asserts_module.test_file_structure(recipe_root)


def test_config_structure():
    """Validate recipe YAML configs against the recipe schema and engine registry."""
    from mvp_engine.engine import ENGINE_REGISTRY
    from mvp_engine.launch import _import_recipe_modules

    recipe_root = _recipe_root()
    _import_recipe_modules(recipe_root)
    config_class = _load_config_class()
    skill_asserts = load_recipe_skill_asserts(recipe_root)

    for config_path in _iter_config_paths(recipe_root):
        config = config_class.model_validate(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))

        try:
            ENGINE_REGISTRY.get(config.engine)
        except KeyError as exc:
            raise AssertionError(
                f"Engine {config.engine} in {config_path.name} is not registered in ENGINE_REGISTRY."
            ) from exc

        for asserts_module in skill_asserts:
            if hasattr(asserts_module, "test_config_structure"):
                asserts_module.test_config_structure(config)


def test_engine_structure():
    """Validate that configured engine classes satisfy the engine interface."""
    from mvp_engine.engine import ENGINE_REGISTRY, Engine
    from mvp_engine.launch import _import_recipe_modules

    recipe_root = _recipe_root()
    _import_recipe_modules(recipe_root)
    config_class = _load_config_class()
    skill_asserts = load_recipe_skill_asserts(recipe_root)

    for config_path in _iter_config_paths(recipe_root):
        config = config_class.model_validate(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
        engine_class = ENGINE_REGISTRY.get(config.engine)
        missing_methods = Engine.__abstractmethods__ & engine_class.__abstractmethods__
        assert not missing_methods, (
            f"Engine {config.engine} in {config_path.name} does not implement abstract methods: "
            f"{sorted(missing_methods)}."
        )

        for asserts_module in skill_asserts:
            if hasattr(asserts_module, "test_engine_structure"):
                asserts_module.test_engine_structure(engine_class)
