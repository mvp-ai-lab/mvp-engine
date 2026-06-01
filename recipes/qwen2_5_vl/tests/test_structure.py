"""Structure and lightweight behavior tests for the qwen2_5_vl recipe."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from mvp_engine.testing.utils import load_recipe_skill_asserts

RECIPE_IMPORT_PATH = "qwen2_5_vl"
CONFIG_SCHEMA_MODULE = f"{RECIPE_IMPORT_PATH}.configs.schema"
CONFIG_CLASS_NAME = "Qwen2_5VLConfig"

EXPECTED_FILES = [
    "README.md",
    "__init__.py",
    "configs/__init__.py",
    "configs/schema.py",
    "configs/stage1.yaml",
    "configs/stage2.yaml",
    "configs/stage3.yaml",
    "engine/__init__.py",
    "engine/qwen2_5_vl_engine.py",
    "guards/__init__.py",
    "guards/loss.py",
    "model/__init__.py",
    "model/qwen2_5_vl.py",
    "model/packing/__init__.py",
    "model/packing/qwen2_5_vl.py",
    "utils/__init__.py",
    "utils/misc.py",
]


def _recipe_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_config_class():
    module = importlib.import_module(CONFIG_SCHEMA_MODULE)
    return getattr(module, CONFIG_CLASS_NAME)


def _iter_config_paths(recipe_root: Path):
    return sorted((recipe_root / "configs").glob("*.yaml"))


def test_file_structure():
    """Validate the intentionally small recipe file layout."""
    recipe_root = _recipe_root()
    for relative_path in EXPECTED_FILES:
        assert (recipe_root / relative_path).exists(), f"{relative_path} does not exist in the recipe root."

    assert not (recipe_root / "dataset").exists(), "image-only recipe should use MLLMDataKit defaults."
    for asserts_module in load_recipe_skill_asserts(recipe_root):
        if hasattr(asserts_module, "test_file_structure"):
            asserts_module.test_file_structure(recipe_root)


def test_config_structure():
    """Validate all YAML configs against the recipe schema and engine registry."""
    from mvp_engine.engine import ENGINE_REGISTRY
    from mvp_engine.launch import _import_recipe_modules

    recipe_root = _recipe_root()
    _import_recipe_modules(recipe_root)
    config_class = _load_config_class()
    skill_asserts = load_recipe_skill_asserts(recipe_root)

    for config_path in _iter_config_paths(recipe_root):
        config = config_class.model_validate(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
        ENGINE_REGISTRY.get(config.engine)
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

    for config_path in _iter_config_paths(recipe_root):
        config = config_class.model_validate(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
        engine_class = ENGINE_REGISTRY.get(config.engine)
        missing_methods = Engine.__abstractmethods__ & engine_class.__abstractmethods__
        assert not missing_methods


def test_packed_position_ids_are_image_only_and_grid_driven():
    """Validate image-only packed position ids without hard-coded patch geometry."""
    from qwen2_5_vl.model.packing import build_qwen2_5_vl_packed_position_ids

    config = SimpleNamespace(
        image_token_id=10,
        video_token_id=11,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    input_ids = torch.tensor([[1, 10, 10, 2, 3, 10, 10, 10, 10, 4]])
    pack_segment_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2, 2, 2]])
    image_grid_thw = torch.tensor([[1, 2, 4], [1, 4, 4]])

    position_ids = build_qwen2_5_vl_packed_position_ids(
        input_ids=input_ids,
        pack_segment_ids=pack_segment_ids,
        image_grid_thw=image_grid_thw,
        model_config=config,
    )

    assert position_ids.shape == (4, 1, input_ids.shape[1])
    assert position_ids[0, 0].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]
    assert position_ids[3, 0, 1:3].tolist() == [1, 2]
    assert position_ids[3, 0, 5:9].tolist() == [1, 2, 1, 2]


def test_packed_position_rejects_image_metadata_mismatch():
    """Validate that image spans must match image_grid_thw."""
    from qwen2_5_vl.model.packing import build_qwen2_5_vl_packed_position_ids

    config = SimpleNamespace(image_token_id=10, vision_config=SimpleNamespace(spatial_merge_size=2))
    with pytest.raises(ValueError, match="Image token span"):
        build_qwen2_5_vl_packed_position_ids(
            input_ids=torch.tensor([[1, 10, 2]]),
            pack_segment_ids=torch.tensor([[1, 1, 1]]),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            model_config=config,
        )


def test_prepare_rejects_video_fields_and_tokens():
    """Validate that this version fails clearly for unsupported video inputs."""
    from qwen2_5_vl.model.packing import prepare_packed_model_inputs

    config = SimpleNamespace(
        image_token_id=10,
        video_token_id=11,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[1, 2]]),
        "pack_segment_ids": torch.tensor([[1, 1]]),
        "second_per_grid_ts": torch.tensor([1.0]),
    }
    with pytest.raises(NotImplementedError, match="image-only"):
        prepare_packed_model_inputs(
            batch,
            model_config=config,
            attn_implementation="eager",
            mask_dtype=torch.float32,
        )

    batch.pop("second_per_grid_ts")
    batch["input_ids"] = torch.tensor([[1, 11]])
    with pytest.raises(NotImplementedError, match="image-only"):
        prepare_packed_model_inputs(
            batch,
            model_config=config,
            attn_implementation="eager",
            mask_dtype=torch.float32,
        )


def test_vision_flops_uses_patch_size_from_config():
    """Validate that vision FLOPs read Qwen2.5-VL geometry from config."""
    from qwen2_5_vl.model.qwen2_5_vl import _calculate_vision_flops

    base = dict(
        spatial_merge_size=2,
        hidden_size=16,
        depth=2,
        intermediate_size=32,
        out_hidden_size=16,
        in_channels=3,
        temporal_patch_size=2,
        window_size=112,
        fullatt_block_indexes=[],
    )
    grid = torch.tensor([[1, 4, 4]])
    flops_14, _ = _calculate_vision_flops(vision_cfg=SimpleNamespace(**base, patch_size=14), image_grid_thw=grid)
    flops_16, _ = _calculate_vision_flops(vision_cfg=SimpleNamespace(**base, patch_size=16), image_grid_thw=grid)
    assert flops_14 != flops_16

