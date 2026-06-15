"""Structure tests for the PanguVL recipe."""

import os
from pathlib import Path
from typing import get_type_hints

from omegaconf import OmegaConf

from mvp_engine.testing.utils import load_recipe_skill_asserts


def test_file_structure():
    """Validate the baseline PanguVL file layout."""
    expected_files = [
        "README.md",
        "__init__.py",
        "configs/__init__.py",
        "configs/schema.py",
        "configs/stage1.yaml",
        "configs/stage2.yaml",
        "configs/stage3.yaml",
        "dataset/__init__.py",
        "dataset/collator.py",
        "dataset/dataset.py",
        "dataset/gate.py",
        "dataset/packing.py",
        "dataset/processor.py",
        "dataset/types.py",
        "engine/__init__.py",
        "engine/panguvl_engine.py",
        "model/__init__.py",
        "model/qwen3_vl.py",
        "model/packing/__init__.py",
        "model/packing/fa2_patch.py",
        "model/packing/prepare.py",
        "model/packing/qwen3_vl.py",
        "third_party/README.md",
        "third_party/configuration_openpangu_vl.py",
        "third_party/imageprocessor_openpangu_vl.py",
        "third_party/modeling_openpangu_embedded.py",
        "third_party/modeling_openpangu_vl.py",
        "utils/__init__.py",
    ]

    recipe_root = Path(__file__).resolve().parent.parent

    for relative_path in expected_files:
        assert (recipe_root / relative_path).exists(), f"{relative_path} does not exist in the recipe root."

    for asserts_module in load_recipe_skill_asserts(recipe_root):
        if hasattr(asserts_module, "test_file_structure"):
            asserts_module.test_file_structure(recipe_root)


def test_config_structure():
    """Validate all PanguVL YAML configs against the recipe schema."""
    from mvp_engine.engine import ENGINE_REGISTRY
    from mvp_engine.launch import _import_recipe_modules
    from recipes.panguvl.configs.schema import PanguvlConfig

    recipe_root = Path(__file__).resolve().parent.parent
    _import_recipe_modules(recipe_root)
    skill_asserts = load_recipe_skill_asserts(recipe_root)

    for config_file in os.listdir(recipe_root / "configs"):
        if not config_file.endswith(".yaml"):
            continue

        config_path = recipe_root / "configs" / config_file
        config = PanguvlConfig.model_validate(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))

        try:
            ENGINE_REGISTRY.get(config.engine)
        except KeyError as exc:
            raise AssertionError(
                f"Engine {config.engine} in {config_file} is not registered in ENGINE_REGISTRY."
            ) from exc

        assert config.data.train_path is None or config.data.train_path.strip()
        assert config.model.pretrained_model_name_or_path.strip()
        assert config.loop.policy == "iter"

        for asserts_module in skill_asserts:
            if hasattr(asserts_module, "test_config_structure"):
                asserts_module.test_config_structure(config)


def test_engine_structure():
    """Validate that configured engine classes satisfy the engine interface."""
    from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
    from mvp_engine.launch import _import_recipe_modules
    from recipes.panguvl.configs.schema import PanguvlConfig

    recipe_root = Path(__file__).resolve().parent.parent
    _import_recipe_modules(recipe_root)
    skill_asserts = load_recipe_skill_asserts(recipe_root)

    for config_file in os.listdir(recipe_root / "configs"):
        if not config_file.endswith(".yaml"):
            continue

        config_path = recipe_root / "configs" / config_file
        config = PanguvlConfig.model_validate(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
        engine_class = ENGINE_REGISTRY.get(config.engine)
        missing_methods = Engine.__abstractmethods__ & engine_class.__abstractmethods__
        assert not missing_methods, (
            f"Engine {config.engine} in {config_file} does not implement abstract methods: {sorted(missing_methods)}."
        )
        for method_name in (
            "train_pre_step",
            "forward_step",
            "backward_step",
            "optimizer_step",
            "train_post_step",
        ):
            type_hints = get_type_hints(getattr(engine_class, method_name))
            assert type_hints.get("ctx") is TrainStepContext

        for asserts_module in skill_asserts:
            if hasattr(asserts_module, "test_engine_structure"):
                asserts_module.test_engine_structure(engine_class)
