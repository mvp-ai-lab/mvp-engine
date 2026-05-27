"""Structure tests for the Basic VLM recipe.

These tests validate recipe files, config schemas, and engine registration
without executing training. Recipe-local skill assertions can extend each layer
by defining matching functions in ``tests/skills/<skill-id>/asserts.py``.
"""

import os
from pathlib import Path

from omegaconf import OmegaConf

from mvp_engine.testing.utils import load_recipe_skill_asserts


def test_file_structure():
    """Validate the baseline Basic VLM file layout.

    Skill assertion modules may add their own file checks by defining
    ``test_file_structure(recipe_root)``. The baseline list stays focused on
    core recipe-owned files so skill-specific files remain owned by skills.
    """
    EXPECTED_FILES = [
        "README.md",
        "__init__.py",
        "configs/__init__.py",
        "configs/schema.py",
        "configs/stage1.yaml",
        "configs/stage2.yaml",
        "configs/stage3.yaml",
        "engine/__init__.py",
        "engine/basic_vlm_engine.py",
        "guards/__init__.py",
        "guards/data.py",
        "guards/loss.py",
        "model/__init__.py",
        "model/qwen3_vl.py",
        "model/packing/__init__.py",
        "model/packing/prepare.py",
        "model/packing/qwen3_vl.py",
        "utils/__init__.py",
        "utils/log/__init__.py",
        "utils/log/mfu.py",
        "utils/misc.py",
    ]

    recipe_root = Path(__file__).resolve().parent.parent

    for relative_path in EXPECTED_FILES:
        assert (recipe_root / relative_path).exists(), f"{relative_path} does not exist in the recipe root."

    for asserts_module in load_recipe_skill_asserts(recipe_root):
        if hasattr(asserts_module, "test_file_structure"):
            asserts_module.test_file_structure(recipe_root)


def test_config_structure():
    """Validate all Basic VLM YAML configs against the recipe schema.

    Each config is parsed through ``BasicVLMConfig`` and its engine name is
    resolved through ``ENGINE_REGISTRY``. Skill assertion modules may add
    config-level checks by defining ``test_config_structure(config)``.
    """
    from mvp_engine.engine import ENGINE_REGISTRY
    from mvp_engine.launch import _import_recipe_modules
    from recipes.basic_vlm.configs.schema import BasicVLMConfig

    _import_recipe_modules(Path(__file__).resolve().parent.parent)

    recipe_root = Path(__file__).resolve().parent.parent
    configs = os.listdir(recipe_root / "configs")
    skill_asserts = load_recipe_skill_asserts(recipe_root)

    try:
        for config_file in configs:
            if config_file.endswith(".yaml"):
                config_path = recipe_root / "configs" / config_file
                config = BasicVLMConfig.model_validate(
                    OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
                )

                try:
                    ENGINE_REGISTRY.get(config.engine)
                except KeyError as exc:
                    raise AssertionError(
                        f"Engine {config.engine} in {config_file} is not registered in ENGINE_REGISTRY."
                    ) from exc
                for asserts_module in skill_asserts:
                    if hasattr(asserts_module, "test_config_structure"):
                        asserts_module.test_config_structure(config)

    except Exception as e:
        raise AssertionError(f"Config structure test failed for {config_file}: {e}")


def test_engine_structure():
    """Validate that configured engine classes satisfy the engine interface.

    The test verifies that each config's engine resolves to a concrete
    ``Engine`` subclass with all abstract methods implemented. Skill assertion
    modules may add engine-level checks by defining
    ``test_engine_structure(engine_class)``.
    """
    from mvp_engine.engine import ENGINE_REGISTRY, Engine
    from mvp_engine.launch import _import_recipe_modules
    from recipes.basic_vlm.configs.schema import BasicVLMConfig

    _import_recipe_modules(Path(__file__).resolve().parent.parent)

    recipe_root = Path(__file__).resolve().parent.parent
    configs = os.listdir(recipe_root / "configs")
    skill_asserts = load_recipe_skill_asserts(recipe_root)

    try:
        for config_file in configs:
            if config_file.endswith(".yaml"):
                config_path = recipe_root / "configs" / config_file
                config = BasicVLMConfig.model_validate(
                    OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
                )

                engine_class = ENGINE_REGISTRY.get(config.engine)
                missing_methods = Engine.__abstractmethods__ & engine_class.__abstractmethods__
                assert not missing_methods, (
                    f"Engine {config.engine} in {config_file} does not implement abstract methods: "
                    f"{sorted(missing_methods)}."
                )
                for asserts_module in skill_asserts:
                    if hasattr(asserts_module, "test_engine_structure"):
                        asserts_module.test_engine_structure(engine_class)

    except Exception as e:
        raise AssertionError(f"Config structure test failed for {config_file}: {e}")
