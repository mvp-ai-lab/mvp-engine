"""Recipe-local assertions for the kit-aware vlm-packing skill.

Copy this file to:
recipes/<recipe>/tests/skills/vlm-packing/asserts.py
"""

import ast
import inspect
import re
import textwrap
from typing import Any

import torch


def test_config_structure(config: Any) -> None:
    """Verify present packing config fields have valid values."""
    if hasattr(config.data, "packing_selection_strategy"):
        assert config.data.packing_selection_strategy in {"random", "best_fit"}, (
            "packing_selection_strategy must be 'random' or 'best_fit'."
        )
    if hasattr(config.data, "packing_open_pack_limit"):
        assert int(config.data.packing_open_pack_limit) >= 1, "packing_open_pack_limit must be >= 1."
    if hasattr(config.data, "packing_buffer_size"):
        assert int(config.data.packing_buffer_size) >= 0, "packing_buffer_size must be >= 0."


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine uses PackingOptions and prepares packed model inputs."""
    source = "\n".join(
        textwrap.dedent(inspect.getsource(method))
        for name in ("prepare_dataloader", "train_pre_step", "forward_step")
        if (method := getattr(engine_class, name, None)) is not None
    )
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    uses_standard_data_kit = "MLLMDataKit" in source or "data_kit.build_dataset" in source
    assert "PackingOptions" in source or not uses_standard_data_kit, (
        "Standard MLLM engines should pass PackingOptions to MLLMDataKit."
    )
    assert "pack_segment_ids" in source or any("prepare_packed" in name for name in calls), (
        "Engine must preserve or prepare packed metadata before model forward."
    )
    assert not uses_standard_data_kit or re.search(r"\bconfig\.data\.packing\b", source) is None, (
        "Engine must not branch on a data.packing enable flag."
    )


def assert_train_pre_step_end(result: Any) -> None:
    """After device transfer, verify packed batches carry valid metadata."""
    result = _extract_batch(result)
    assert isinstance(result, dict), "train_pre_step must return a batch dictionary."
    assert "source_sample_num" in result, "Packed batches must include source_sample_num."
    assert torch.is_tensor(result["source_sample_num"]), "source_sample_num must be a tensor after collation."

    if "pack_segment_ids" in result:
        segment_ids = result["pack_segment_ids"]
        assert segment_ids.shape == result["input_ids"].shape, "pack_segment_ids must match input_ids shape."
        assert int(segment_ids.max().item()) >= 1, "Packed batch must contain at least one active segment."


def _extract_batch(result: Any) -> Any:
    """Support hooks that pass either the batch or a TrainStepContext."""
    return getattr(result, "data", result)
