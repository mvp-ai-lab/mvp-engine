"""Recipe-local assertions for the vlm-data-pipeline skill.

Copy this file to:
recipes/<recipe>/tests/skills/vlm-data-pipeline/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Any

import torch


def test_file_structure(recipe_root: Path) -> None:
    """Verify the recipe owns the main VLM data-pipeline layers."""
    asserts_path = recipe_root / "tests" / "skills" / "vlm-data-pipeline" / "asserts.py"
    dataset_dir = recipe_root / "dataset"
    expected_files = {
        "dataset.py": dataset_dir / "dataset.py",
        "preprocess.py": dataset_dir / "preprocess.py",
        "processor.py": dataset_dir / "processor.py",
        "collator.py": dataset_dir / "collator.py",
        "types.py": dataset_dir / "types.py",
    }

    assert asserts_path.exists(), f"Missing recipe-local skill assertions: {asserts_path}"
    for name, path in expected_files.items():
        assert path.exists(), f"VLM data pipeline must include dataset/{name}."

    dataset_source = expected_files["dataset.py"].read_text(encoding="utf-8")
    preprocess_source = expected_files["preprocess.py"].read_text(encoding="utf-8")
    processor_source = expected_files["processor.py"].read_text(encoding="utf-8")
    collator_source = expected_files["collator.py"].read_text(encoding="utf-8")

    assert "build_dataset" in dataset_source, "dataset.py must expose build_dataset(...)."
    assert "process_sample" in preprocess_source, "preprocess.py must expose process_sample(...)."
    assert "input_ids" in preprocess_source, "preprocess.py must produce input_ids."
    assert "attention_mask" in preprocess_source, "preprocess.py must produce attention_mask."
    assert "labels" in preprocess_source, "preprocess.py must produce labels."
    assert "images" in preprocess_source or "pixel_values" in preprocess_source, (
        "preprocess.py must handle media refs or media tensors."
    )
    assert "AutoProcessor" in processor_source or "Processor" in processor_source, (
        "processor.py must own target processor construction."
    )
    assert "pad_sequence" in collator_source or ".pad(" in collator_source, "collator.py must pad token tensors."
    assert "pixel_values" in collator_source or "image_grid" in collator_source or "video" in collator_source, (
        "collator.py must handle model media tensors or adapt this assertion for a text-only recipe."
    )


def test_config_structure(config: Any) -> None:
    """Verify config exposes the basic data and model fields needed by VLM loaders."""
    assert hasattr(config, "data"), "Config must expose a data section."
    assert hasattr(config, "model"), "Config must expose a model section."
    assert hasattr(config.data, "train_path"), "config.data.train_path is required."
    assert hasattr(config.data, "max_seq_len"), "config.data.max_seq_len is required."
    assert int(config.data.max_seq_len) > 0, "config.data.max_seq_len must be positive."
    assert hasattr(config.data, "batch_size"), "config.data.batch_size is required."
    assert int(config.data.batch_size) > 0, "config.data.batch_size must be positive."
    assert hasattr(config.model, "pretrained_model_name_or_path"), (
        "config.model.pretrained_model_name_or_path is required for processor/model loading."
    )


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine wires processor, dataset, and collator together."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_dataloader))
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert "processor" in source, "prepare_dataloader must construct or reuse the model processor."
    assert any(name.endswith("build_dataset") for name in calls), "prepare_dataloader must call build_dataset(...)."
    assert "collate" in source.lower() or "Collator" in source, "prepare_dataloader must install a VLM collator."


def assert_train_pre_step_end(result: dict[str, Any]) -> None:
    """After device transfer, verify the real batch satisfies the VLM tensor contract."""
    assert isinstance(result, dict), "train_pre_step must return a batch dictionary."
    for key in ("input_ids", "attention_mask", "labels"):
        assert key in result, f"Batch missing required key: {key}."
        assert torch.is_tensor(result[key]), f"Batch key {key} must be a tensor."

    input_ids = result["input_ids"]
    attention_mask = result["attention_mask"]
    labels = result["labels"]

    assert input_ids.ndim == 2, f"input_ids must be 2D, got shape {tuple(input_ids.shape)}."
    assert attention_mask.shape == input_ids.shape, "attention_mask must match input_ids shape."
    assert labels.shape == input_ids.shape, "labels must match input_ids shape."
    assert int(attention_mask.sum().item()) > 0, "Batch attention_mask must contain at least one valid token."

    has_images = "pixel_values" in result and result["pixel_values"] is not None
    has_videos = "pixel_values_videos" in result and result["pixel_values_videos"] is not None
    if has_images:
        assert "image_grid_thw" in result, "Image batches must include image_grid_thw."
    if has_videos:
        assert "video_grid_thw" in result, "Video batches must include video_grid_thw."
