"""Qwen3-VL Liger module support tests."""

from __future__ import annotations

import pytest

from recipes.qwen3_vl.configs.schema import Qwen3VLLigerKernelConfig
from recipes.qwen3_vl.model.liger import _resolve_modules


def test_auto_modules_do_not_enable_swiglu() -> None:
    """Qwen3-VL auto mode must not report unsupported SwiGLU as enabled."""
    modules, explicit_modules = _resolve_modules(Qwen3VLLigerKernelConfig(modules="auto"), stage="pre_build")

    assert explicit_modules is False
    assert modules["swiglu"] is False


def test_explicit_swiglu_enabled_is_rejected() -> None:
    """Explicit unsupported Qwen3-VL SwiGLU requests must fail fast."""
    config = Qwen3VLLigerKernelConfig(modules={"swiglu": True})

    with pytest.raises(ValueError, match="swiglu"):
        _resolve_modules(config, stage="pre_build")
