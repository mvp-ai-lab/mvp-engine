"""Recipe-local assertions for the vlm-packing skill.

Copy this file to:
recipes/<recipe>/tests/skills/vlm-packing/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Any

import torch


def test_file_structure(recipe_root: Path) -> None:
    """Verify the recipe owns VLM packing layers."""
    asserts_path = recipe_root / "tests" / "skills" / "vlm-packing" / "asserts.py"
    required_files = {
        "dataset/packing.py": recipe_root / "dataset" / "packing.py",
        "dataset/collator.py": recipe_root / "dataset" / "collator.py",
        "dataset/dataset.py": recipe_root / "dataset" / "dataset.py",
    }

    assert asserts_path.exists(), f"Missing recipe-local skill assertions: {asserts_path}"
    for name, path in required_files.items():
        assert path.exists(), f"VLM packing requires {name}."

    packing_source = required_files["dataset/packing.py"].read_text(encoding="utf-8")
    collator_source = required_files["dataset/collator.py"].read_text(encoding="utf-8")
    dataset_source = required_files["dataset/dataset.py"].read_text(encoding="utf-8")
    model_packing_dir = recipe_root / "model" / "packing"

    assert "pack_segment" in packing_source, "dataset/packing.py must build packed segment metadata."
    assert "source_sample_num" in packing_source, "Packed samples must record source_sample_num or equivalent."
    assert "max_seq_len" in packing_source, "Packer must respect max_seq_len."
    assert "packing_selection_strategy" in dataset_source or "build_packed" in dataset_source, (
        "dataset.py must wire the packing stage into the dataset lifecycle."
    )
    assert "pack_segment" in collator_source, "collator.py must pad packed segment metadata."
    assert "Packed and unpacked" in collator_source or "mixed" in collator_source.lower(), (
        "collator.py must reject or explicitly handle mixed packed/unpacked batches."
    )
    assert model_packing_dir.exists(), "VLM packing requires model/packing/ preparation helpers."
    assert any(path.name == "prepare.py" for path in model_packing_dir.glob("*.py")), (
        "model/packing must include packed model-input preparation."
    )


def test_config_structure(config: Any) -> None:
    """Verify packing config exposes the standard knobs."""
    assert hasattr(config.data, "packing"), "config.data.packing is required."
    assert isinstance(config.data.packing, bool), "config.data.packing must be a bool."
    assert hasattr(config.data, "packing_selection_strategy"), "config.data.packing_selection_strategy is required."
    assert config.data.packing_selection_strategy in {"random", "best_fit"}, (
        "packing_selection_strategy must be 'random' or 'best_fit'."
    )
    assert int(config.data.packing_open_pack_limit) >= 1, "packing_open_pack_limit must be >= 1."
    assert int(config.data.packing_buffer_size) >= 0, "packing_buffer_size must be >= 0."
    assert int(config.data.max_seq_len) > 0, "max_seq_len must be positive."


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine prepares packed model inputs before forward."""
    source = "\n".join(
        textwrap.dedent(inspect.getsource(method))
        for name in ("prepare_model", "train_pre_step", "forward_step", "train_one_step")
        if (method := getattr(engine_class, name, None)) is not None
    )
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert "config.data.packing" in source or "self.config.data.packing" in source, (
        "Engine must branch on config.data.packing for packed model-input preparation."
    )
    assert any("prepare_packed" in name for name in calls) or "pack_segment_ids" in source, (
        "Engine must prepare packed metadata before model forward."
    )


def assert_train_pre_step_end(engine, result: dict[str, Any]) -> None:
    """After device transfer, verify packed batches carry valid metadata."""
    if not getattr(engine.config.data, "packing", False):
        return

    assert isinstance(result, dict), "train_pre_step must return a batch dictionary."
    assert "source_sample_num" in result, "Packed batches must include source_sample_num."
    assert torch.is_tensor(result["source_sample_num"]), "source_sample_num must be a tensor after collation."

    if "pack_segment_ids" in result:
        segment_ids = result["pack_segment_ids"]
        assert segment_ids.shape == result["input_ids"].shape, "pack_segment_ids must match input_ids shape."
        assert int(segment_ids.max().item()) >= 1, "Packed batch must contain at least one active segment."
    else:
        assert "position_ids" in result, "Packed batch must be converted to position_ids before forward."
        assert "attention_mask" in result, "Packed batch must keep or build an attention_mask before forward."
