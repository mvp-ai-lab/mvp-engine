"""Unit tests for OpenPangu mRoPE inspection helpers."""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_rope_module():
    recipe_root = Path(__file__).resolve().parent.parent
    module_name = "_panguvl_test_inspect_rope_config"
    spec = importlib.util.spec_from_file_location(module_name, recipe_root / "tools" / "inspect_rope_config.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


rope = _load_rope_module()


class _RotaryEmbedding:
    mrope_section = [10, 27, 27]
    mrope_interleaved = True
    mrope_dim = list(range(64))


class _RotaryEmbeddingMissingDim:
    mrope_section = [10, 27, 27]
    mrope_interleaved = True


def test_mrope_scaling_accepts_openpangu_default_fields():
    rope.assert_mrope_scaling(
        {
            "rope_type": "default",
            "type": "default",
            "mrope_section": [10, 27, 27],
            "mrope_interleaved": True,
        },
        source="test",
    )


def test_mrope_scaling_rejects_missing_interleaved_flag():
    with pytest.raises(RuntimeError, match="mrope_interleaved"):
        rope.assert_mrope_scaling(
            {
                "rope_type": "default",
                "mrope_section": [10, 27, 27],
                "mrope_interleaved": False,
            },
            source="test",
        )


def test_rotary_mrope_active_requires_interleaved_dimension_mapping():
    rope.assert_rotary_mrope_active(_RotaryEmbedding())

    with pytest.raises(RuntimeError, match="mrope_dim"):
        rope.assert_rotary_mrope_active(_RotaryEmbeddingMissingDim())
