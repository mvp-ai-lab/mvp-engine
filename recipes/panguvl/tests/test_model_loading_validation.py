"""Checkpoint loading validation for the PanguVL model helper."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _install_qwen3_vl_stub() -> None:
    package_names = [
        "transformers",
        "transformers.models",
        "transformers.models.qwen3_vl",
    ]
    for package_name in package_names:
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package

    transformers = sys.modules["transformers"]

    class _AutoModelForCausalLM:
        pass

    transformers.AutoModelForCausalLM = _AutoModelForCausalLM

    module_name = "transformers.models.qwen3_vl.modeling_qwen3_vl"
    module = types.ModuleType(module_name)

    class Qwen3VLCausalLMOutputWithPast:
        pass

    module.Qwen3VLCausalLMOutputWithPast = Qwen3VLCausalLMOutputWithPast
    sys.modules[module_name] = module


def _load_qwen3_vl_module():
    _install_qwen3_vl_stub()
    recipe_root = Path(__file__).resolve().parent.parent
    module_name = "_panguvl_test_qwen3_vl"
    spec = importlib.util.spec_from_file_location(module_name, recipe_root / "model" / "qwen3_vl.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


qwen3_vl = _load_qwen3_vl_module()


def test_checkpoint_loading_validation_rejects_missing_language_model_weights():
    loading_info = {
        "missing_keys": {"model.language_model.layers.0.self_attn.q_proj.weight"},
        "unexpected_keys": {"model.layers.0.self_attn.q_proj.weight"},
        "mismatched_keys": set(),
        "conversion_errors": {},
    }

    with pytest.raises(RuntimeError, match="checkpoint load left pretrained weights"):
        qwen3_vl._validate_checkpoint_loading_info(loading_info)


def test_checkpoint_loading_validation_accepts_clean_loading_info():
    qwen3_vl._validate_checkpoint_loading_info(
        {
            "missing_keys": set(),
            "unexpected_keys": set(),
            "mismatched_keys": set(),
            "conversion_errors": {},
        }
    )
