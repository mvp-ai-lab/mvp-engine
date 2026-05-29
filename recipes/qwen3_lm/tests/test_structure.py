"""Structure tests for the Qwen3 LM recipe."""

from pathlib import Path

from omegaconf import OmegaConf

from mvp_engine.testing.utils import load_recipe_skill_asserts

EXPECTED_FILES = [
    "README.md",
    "__init__.py",
    "configs/__init__.py",
    "configs/schema.py",
    "configs/train.yaml",
    "configs/smoke.yaml",
    "dataset/__init__.py",
    "dataset/collator.py",
    "dataset/dataset.py",
    "dataset/packing.py",
    "dataset/preprocess.py",
    "dataset/processor.py",
    "dataset/types.py",
    "engine/__init__.py",
    "engine/qwen3_lm_engine.py",
    "guards/__init__.py",
    "guards/data.py",
    "guards/loss.py",
    "model/__init__.py",
    "model/qwen3.py",
    "model/packing/__init__.py",
    "model/packing/fa2_patch.py",
    "model/packing/prepare.py",
    "utils/__init__.py",
    "utils/log/__init__.py",
    "utils/log/mfu.py",
    "utils/misc.py",
]


def _recipe_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _iter_config_paths(recipe_root: Path):
    return sorted((recipe_root / "configs").glob("*.yaml"))


def test_file_structure():
    """Validate the baseline Qwen3 LM file layout."""
    recipe_root = _recipe_root()
    for relative_path in EXPECTED_FILES:
        assert (recipe_root / relative_path).exists(), f"{relative_path} does not exist in the recipe root."

    for asserts_module in load_recipe_skill_asserts(recipe_root):
        if hasattr(asserts_module, "test_file_structure"):
            asserts_module.test_file_structure(recipe_root)


def test_config_structure():
    """Validate all Qwen3 LM YAML configs against the recipe schema."""
    from mvp_engine.engine import ENGINE_REGISTRY
    from mvp_engine.launch import _import_recipe_modules
    from recipes.qwen3_lm.configs.schema import Qwen3LMConfig

    recipe_root = _recipe_root()
    _import_recipe_modules(recipe_root)
    skill_asserts = load_recipe_skill_asserts(recipe_root)

    for config_path in _iter_config_paths(recipe_root):
        config = Qwen3LMConfig.model_validate(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))

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
    from recipes.qwen3_lm.configs.schema import Qwen3LMConfig

    recipe_root = _recipe_root()
    _import_recipe_modules(recipe_root)
    skill_asserts = load_recipe_skill_asserts(recipe_root)

    for config_path in _iter_config_paths(recipe_root):
        config = Qwen3LMConfig.model_validate(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
        engine_class = ENGINE_REGISTRY.get(config.engine)
        missing_methods = Engine.__abstractmethods__ & engine_class.__abstractmethods__
        assert not missing_methods, (
            f"Engine {config.engine} in {config_path.name} does not implement abstract methods: "
            f"{sorted(missing_methods)}."
        )

        for asserts_module in skill_asserts:
            if hasattr(asserts_module, "test_engine_structure"):
                asserts_module.test_engine_structure(engine_class)
