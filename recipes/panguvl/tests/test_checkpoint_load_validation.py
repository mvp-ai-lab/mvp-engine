"""Unit tests for PanguVL checkpoint-load validation helpers."""

import importlib.util
import json
import sys
from pathlib import Path


def _load_validator_module():
    recipe_root = Path(__file__).resolve().parent.parent
    module_name = "_panguvl_validate_checkpoint_load"
    spec = importlib.util.spec_from_file_location(
        module_name,
        recipe_root / "tools" / "validate_checkpoint_load.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator_module()


def test_openpangu_key_remap_matches_runtime_model_namespace():
    assert (
        validator.remap_openpangu_key("model.layers.0.self_attn.q_proj.weight")
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert validator.remap_openpangu_key("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert validator.remap_openpangu_key("lm_head.weight") == "lm_head.weight"
    assert validator.remap_openpangu_key("visual.blocks.0.norm1.weight") == "model.visual.blocks.0.norm1.weight"
    assert validator.remap_openpangu_key("model.visual.blocks.0.norm1.weight") == "model.visual.blocks.0.norm1.weight"


def test_metadata_detects_checkpoint_that_requires_openpangu_remap(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "openpangu_vl",
                "architectures": ["OpenPanguVLForConditionalGeneration"],
                "auto_map": {"AutoModelForCausalLM": "modeling_openpangu_vl.OpenPanguVL"},
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.layers.0.input_layernorm.weight": "model.safetensors",
                    "model.embed_tokens.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    keys = validator.load_checkpoint_keys(checkpoint_dir)

    assert validator.requires_openpangu_remap(keys)
    remapped = validator.remap_checkpoint_keys(keys)
    assert "model.language_model.layers.0.input_layernorm.weight" in remapped
    assert "model.language_model.embed_tokens.weight" in remapped
    assert "lm_head.weight" in remapped
