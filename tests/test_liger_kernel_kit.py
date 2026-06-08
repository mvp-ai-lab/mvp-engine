"""Tests for reusable Liger Kernel kit helpers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from mvp_engine.kit import LigerKernelKit, LigerKernelReport


class FakeRMSNorm(torch.nn.Module):
    """Minimal source RMSNorm-like module."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps


class FakeLigerRMSNorm(torch.nn.Module):
    """Minimal Liger RMSNorm replacement."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps


class FakeLigerLayerNorm(torch.nn.LayerNorm):
    """Minimal Liger LayerNorm replacement."""


def install_fake_liger(monkeypatch: pytest.MonkeyPatch, patch_calls: list[dict]) -> ModuleType:
    """Install fake liger_kernel modules in sys.modules."""
    package = ModuleType("liger_kernel")
    package.__path__ = []
    transformers = ModuleType("liger_kernel.transformers")

    def apply_liger_kernel_to_qwen2(
        rms_norm: bool = False,
        rope: bool = False,
        swiglu: bool = False,
        fused_linear_cross_entropy: bool = True,
    ) -> None:
        patch_calls.append(
            {
                "helper": "qwen2",
                "rms_norm": rms_norm,
                "rope": rope,
                "swiglu": swiglu,
                "fused_linear_cross_entropy": fused_linear_cross_entropy,
            }
        )

    def apply_liger_kernel_to_qwen3_vl(
        rms_norm: bool = False,
        rope: bool = False,
        swiglu: bool = False,
        fused_linear_cross_entropy: bool = True,
    ) -> None:
        patch_calls.append(
            {
                "rms_norm": rms_norm,
                "rope": rope,
                "swiglu": swiglu,
                "fused_linear_cross_entropy": fused_linear_cross_entropy,
            }
        )

    transformers.apply_liger_kernel_to_qwen2 = apply_liger_kernel_to_qwen2
    transformers.apply_liger_kernel_to_qwen3_vl = apply_liger_kernel_to_qwen3_vl
    transformers.LigerRMSNorm = FakeLigerRMSNorm
    transformers.LigerLayerNorm = FakeLigerLayerNorm
    monkeypatch.setitem(sys.modules, "liger_kernel", package)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers)
    return transformers


def install_fake_auto_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_type: str | None,
    calls: list[dict] | None = None,
) -> None:
    """Install a fake transformers.AutoConfig in sys.modules."""
    transformers = ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name_or_path: str, **kwargs) -> SimpleNamespace:
            if calls is not None:
                calls.append({"model_name_or_path": model_name_or_path, **kwargs})
            return SimpleNamespace(model_type=model_type)

    transformers.AutoConfig = FakeAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_importing_liger_kernel_kit_does_not_require_liger_package() -> None:
    """Kit import should be safe when optional liger-kernel is absent."""
    assert LigerKernelKit().__class__.__name__ == "LigerKernelKit"


def test_auto_modules_do_not_enable_qwen3_vl_swiglu() -> None:
    """Qwen3-VL auto mode must not report unsupported SwiGLU as enabled."""
    modules = LigerKernelKit().resolve_modules(stage="pre_build", model_family="qwen3_vl", modules="auto")

    assert modules["swiglu"] is False
    assert modules["rms_norm"] is True


def test_explicit_qwen3_vl_swiglu_enabled_is_rejected() -> None:
    """Explicit unsupported Qwen3-VL SwiGLU requests must fail fast."""
    with pytest.raises(ValueError, match="swiglu"):
        LigerKernelKit().resolve_modules(
            stage="pre_build",
            model_family="qwen3_vl",
            modules={"swiglu": True},
        )


def test_pre_build_dispatch_passes_disabled_loss_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-build dispatch should override Liger defaults with explicit False values."""
    patch_calls: list[dict] = []
    install_fake_liger(monkeypatch, patch_calls)

    report = LigerKernelKit().apply_pre_build(model_family="qwen3_vl", modules="auto")

    assert isinstance(report, LigerKernelReport)
    assert report.helper == "apply_liger_kernel_to_qwen3_vl"
    assert patch_calls == [
        {
            "rms_norm": True,
            "rope": True,
            "swiglu": False,
            "fused_linear_cross_entropy": False,
        }
    ]


def test_pre_build_infers_model_family_from_hf_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-build dispatch should use HF config model_type when no override is provided."""
    config_calls: list[dict] = []
    patch_calls: list[dict] = []
    install_fake_auto_config(monkeypatch, model_type="qwen3-vl", calls=config_calls)
    install_fake_liger(monkeypatch, patch_calls)

    report = LigerKernelKit().apply_pre_build(model_name_or_path="fake-qwen3-vl", modules="auto")

    assert report.model_family == "qwen3_vl"
    assert report.helper == "apply_liger_kernel_to_qwen3_vl"
    assert config_calls == [{"model_name_or_path": "fake-qwen3-vl", "trust_remote_code": True}]
    assert patch_calls == [
        {
            "rms_norm": True,
            "rope": True,
            "swiglu": False,
            "fused_linear_cross_entropy": False,
        }
    ]


def test_pre_build_uses_model_type_alias_for_helper_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dynamic pre-build dispatch should try a small alias set after the inferred model_type."""
    patch_calls: list[dict] = []
    install_fake_auto_config(monkeypatch, model_type="qwq")
    install_fake_liger(monkeypatch, patch_calls)

    report = LigerKernelKit().apply_pre_build(model_name_or_path="fake-qwq", modules={"rms_norm": True})

    assert report.model_family == "qwq"
    assert report.helper == "apply_liger_kernel_to_qwen2"
    assert patch_calls == [
        {
            "helper": "qwen2",
            "rms_norm": True,
            "rope": False,
            "swiglu": False,
            "fused_linear_cross_entropy": False,
        }
    ]


def test_post_build_replaces_norm_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Post-build replacement should preserve state dict keys and weight values."""
    install_fake_liger(monkeypatch, [])
    model = torch.nn.Sequential(
        torch.nn.ModuleDict(
            {
                "rms": FakeRMSNorm(4, eps=1e-4),
                "ln": torch.nn.LayerNorm(4),
            }
        )
    )
    state_keys_before = sorted(model.state_dict())
    rms_weight_before = model[0]["rms"].weight.detach().clone()

    patched = LigerKernelKit().apply_post_build(model, modules={"rms_norm": True, "layer_norm": True})

    assert patched is model
    assert sorted(model.state_dict()) == state_keys_before
    assert isinstance(model[0]["rms"], FakeLigerRMSNorm)
    assert isinstance(model[0]["ln"], FakeLigerLayerNorm)
    assert torch.equal(model[0]["rms"].weight, rms_weight_before)
    report = model._mvp_engine_liger_kernel  # noqa: SLF001
    replacement_paths = {replacement.path for replacement in report.replacements}
    assert replacement_paths == {"0.rms", "0.ln"}


def test_post_build_infers_model_family_from_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Post-build replacement should record model.config.model_type when no override is provided."""
    install_fake_liger(monkeypatch, [])
    model = torch.nn.Sequential(FakeRMSNorm(4, eps=1e-4))
    model.config = SimpleNamespace(model_type="qwen3-vl")

    LigerKernelKit().apply_post_build(model, modules="auto")

    report = model._mvp_engine_liger_kernel  # noqa: SLF001
    assert report.model_family == "qwen3_vl"
    assert report.modules["rms_norm"] is True
    assert report.modules["swiglu"] is False


def test_post_build_rejects_enabled_module_without_replacer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict post-build mode should fail on enabled modules without replacement support."""
    install_fake_liger(monkeypatch, [])

    with pytest.raises(ValueError, match="rope"):
        LigerKernelKit().apply_post_build(torch.nn.Linear(2, 2), modules={"rope": True})
