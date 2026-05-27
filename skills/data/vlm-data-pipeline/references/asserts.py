"""Recipe-local assertions for the kit-aware vlm-data-pipeline skill.

Copy this file to:
recipes/<recipe>/tests/skills/vlm-data-pipeline/asserts.py
"""

import ast
import inspect
import textwrap
from typing import Any

import torch


def test_config_structure(config: Any) -> None:
    """Verify config exposes data and model sections for recipe wiring."""
    assert hasattr(config, "data"), "Config must expose a data section."
    assert hasattr(config, "model"), "Config must expose a model section."


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine wires the standard MLLM data kit."""
    source = textwrap.dedent(inspect.getsource(engine_class))
    prepare_dataloader = textwrap.dedent(inspect.getsource(engine_class.prepare_dataloader))
    tree = ast.parse(prepare_dataloader)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert "MLLMDataKit" in source or "data_kit" in source, "Recipe should use MLLMDataKit or a subclass."
    assert any(name.endswith("build_processor") for name in calls), "prepare_dataloader must build a processor."
    assert any(name.endswith("build_dataset") for name in calls), "prepare_dataloader must call build_dataset(...)."
    assert any(name.endswith("build_collator") for name in calls), "prepare_dataloader must build the collator."
    assert any(name.endswith("build_dataloader") for name in calls), "prepare_dataloader must build the dataloader."


def assert_train_pre_step_end(result: Any) -> None:
    """After device transfer, verify the real batch satisfies the MLLM tensor contract."""
    result = _extract_batch(result)
    assert isinstance(result, dict), "train_pre_step must return a batch dictionary."
    for key in ("input_ids", "attention_mask", "labels"):
        assert key in result, f"Batch missing required key: {key}."
        assert torch.is_tensor(result[key]), f"Batch key {key} must be a tensor."

    input_ids = result["input_ids"]
    assert input_ids.ndim == 2, f"input_ids must be 2D, got shape {tuple(input_ids.shape)}."
    assert result["attention_mask"].shape == input_ids.shape, "attention_mask must match input_ids shape."
    assert result["labels"].shape == input_ids.shape, "labels must match input_ids shape."
    assert int(result["attention_mask"].sum().item()) > 0, "Batch must contain at least one valid token."

    if result.get("pixel_values") is not None:
        assert "image_grid_thw" in result, "Image batches must include image_grid_thw."
    if result.get("pixel_values_videos") is not None:
        assert "video_grid_thw" in result, "Video batches must include video_grid_thw."


def _extract_batch(result: Any) -> Any:
    """Support hooks that pass either the batch or a TrainStepContext."""
    return getattr(result, "data", result)
