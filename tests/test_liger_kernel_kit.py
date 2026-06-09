"""Tests for the reusable Liger Kernel kit (pre-build only)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from mvp_engine.kit import LigerKernelKit, LigerKernelReport, LigerPatch


def install_fake_liger(monkeypatch: pytest.MonkeyPatch, calls: list[dict]) -> None:
    """Install fake liger_kernel.transformers helpers that record their call kwargs."""
    package = ModuleType("liger_kernel")
    package.__path__ = []
    transformers = ModuleType("liger_kernel.transformers")

    def make_helper(name: str):
        def helper(rope=True, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=True):
            calls.append(
                {
                    "helper": name,
                    "rope": rope,
                    "cross_entropy": cross_entropy,
                    "fused_linear_cross_entropy": fused_linear_cross_entropy,
                    "rms_norm": rms_norm,
                    "swiglu": swiglu,
                }
            )

        return helper

    transformers.apply_liger_kernel_to_qwen2 = make_helper("qwen2")
    transformers.apply_liger_kernel_to_qwen3_vl = make_helper("qwen3_vl")
    monkeypatch.setitem(sys.modules, "liger_kernel", package)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers)


def install_fake_auto_config(monkeypatch: pytest.MonkeyPatch, model_type: str | None) -> None:
    """Install a fake transformers.AutoConfig returning a fixed model_type."""
    transformers = ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name_or_path: str, **kwargs) -> SimpleNamespace:
            return SimpleNamespace(model_type=model_type)

    transformers.AutoConfig = FakeAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def install_fake_custom_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install a fake custom modeling module with swappable symbols."""
    package = ModuleType("fake_custom")
    package.__path__ = []
    modeling = ModuleType("fake_custom.modeling_custom")
    modeling.CustomRMSNorm = type("CustomRMSNorm", (), {})
    modeling.CustomMLP = type("CustomMLP", (), {})
    modeling.apply_rotary_pos_emb = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "fake_custom", package)
    monkeypatch.setitem(sys.modules, "fake_custom.modeling_custom", modeling)
    return modeling


def custom_patch(attr: str) -> LigerPatch:
    return LigerPatch(module="fake_custom.modeling_custom", attr=attr, replacement=object())


def test_importing_kit_does_not_require_liger_package() -> None:
    """Kit import must be safe when the optional liger-kernel package is absent."""
    assert LigerKernelKit().__class__.__name__ == "LigerKernelKit"


def test_official_auto_only_forces_loss_off_and_defers_to_liger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto must pass only explicit loss=False and let liger default rope/norm/mlp."""
    calls: list[dict] = []
    install_fake_liger(monkeypatch, calls)

    report = LigerKernelKit().apply_pre_build(model_family="qwen3-vl", modules="auto")

    assert isinstance(report, LigerKernelReport)
    assert report.route == "official"
    assert report.helper == "apply_liger_kernel_to_qwen3_vl"
    assert report.applied == {"cross_entropy": False, "fused_linear_cross_entropy": False}
    # liger received loss off, while rope/rms_norm/swiglu kept their library defaults (True)
    assert calls == [
        {
            "helper": "qwen3_vl",
            "rope": True,
            "cross_entropy": False,
            "fused_linear_cross_entropy": False,
            "rms_norm": True,
            "swiglu": True,
        }
    ]


def test_official_loss_allowed_auto_passes_no_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """With loss allowed, auto forwards nothing and relies fully on liger defaults."""
    calls: list[dict] = []
    install_fake_liger(monkeypatch, calls)

    report = LigerKernelKit().apply_pre_build(model_family="qwen2", modules="auto", loss_kernels_allowed=True)

    assert report.applied == {}
    assert calls[0]["fused_linear_cross_entropy"] is True  # liger default left untouched


def test_official_explicit_modules_force_loss_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit module flags are forwarded with loss still disabled by default."""
    calls: list[dict] = []
    install_fake_liger(monkeypatch, calls)

    report = LigerKernelKit().apply_pre_build(model_family="qwen2", modules={"rms_norm": True, "rope": False})

    assert report.applied == {
        "rms_norm": True,
        "rope": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": False,
    }


def test_official_infers_family_from_hf_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Family is inferred from AutoConfig.model_type when no override is given."""
    install_fake_auto_config(monkeypatch, model_type="Qwen2")
    install_fake_liger(monkeypatch, [])

    report = LigerKernelKit().apply_pre_build(model_name_or_path="fake-qwen2", modules="auto")

    assert report.model_family == "qwen2"
    assert report.helper == "apply_liger_kernel_to_qwen2"


def test_official_uses_alias_for_helper_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """An aliased model_type (qwq) resolves to the base family helper (qwen2)."""
    install_fake_liger(monkeypatch, [])

    report = LigerKernelKit().apply_pre_build(model_family="qwq", modules="auto")

    assert report.model_family == "qwq"
    assert report.helper == "apply_liger_kernel_to_qwen2"


def test_official_rejects_module_absent_from_helper_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabling a module the official helper does not accept must fail fast."""
    install_fake_liger(monkeypatch, [])

    with pytest.raises(ValueError, match="geglu"):
        LigerKernelKit().apply_pre_build(model_family="qwen2", modules={"geglu": True})


def test_custom_route_swaps_symbols_in_target_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom route setattr-swaps each declared symbol and reports the patched paths."""
    modeling = install_fake_custom_module(monkeypatch)
    patches = {"rms_norm": custom_patch("CustomRMSNorm"), "rope": custom_patch("apply_rotary_pos_emb")}

    report = LigerKernelKit().apply_pre_build(model_family="custom_vlm", custom_patches=patches)

    assert report.route == "custom"
    assert modeling.CustomRMSNorm is patches["rms_norm"].replacement
    assert modeling.apply_rotary_pos_emb is patches["rope"].replacement
    assert set(report.patched) == {
        "fake_custom.modeling_custom.CustomRMSNorm",
        "fake_custom.modeling_custom.apply_rotary_pos_emb",
    }


def test_custom_route_explicit_modules_select_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit module dict applies only the enabled patches."""
    modeling = install_fake_custom_module(monkeypatch)
    original_rope = modeling.apply_rotary_pos_emb
    patches = {"rms_norm": custom_patch("CustomRMSNorm"), "rope": custom_patch("apply_rotary_pos_emb")}

    report = LigerKernelKit().apply_pre_build(custom_patches=patches, modules={"rms_norm": True, "rope": False})

    assert report.patched == ("fake_custom.modeling_custom.CustomRMSNorm",)
    assert modeling.apply_rotary_pos_emb is original_rope  # untouched


def test_custom_route_missing_symbol_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A patch targeting a non-existent symbol fails loudly (e.g. upstream rename)."""
    install_fake_custom_module(monkeypatch)
    patches = {"rms_norm": custom_patch("NoSuchSymbol")}

    with pytest.raises(AttributeError, match="NoSuchSymbol"):
        LigerKernelKit().apply_pre_build(custom_patches=patches)


def test_custom_route_requires_patch_for_enabled_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabling a module without a corresponding custom patch fails fast."""
    install_fake_custom_module(monkeypatch)
    patches = {"rms_norm": custom_patch("CustomRMSNorm")}

    with pytest.raises(ValueError, match="No custom_patches"):
        LigerKernelKit().apply_pre_build(custom_patches=patches, modules={"swiglu": True})


def test_loss_kernels_are_guarded_on_custom_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loss kernels stay disabled unless explicitly allowed by the recipe."""
    install_fake_custom_module(monkeypatch)
    patches = {"fused_linear_cross_entropy": custom_patch("CustomMLP")}

    with pytest.raises(ValueError, match="loss_kernels_allowed"):
        LigerKernelKit().apply_pre_build(custom_patches=patches)


def test_unknown_module_name_is_rejected() -> None:
    """Unsupported semantic module names are rejected before any patching."""
    patches = {"bogus": LigerPatch(module="m", attr="a", replacement=object())}

    with pytest.raises(ValueError, match="Unsupported Liger module"):
        LigerKernelKit().apply_pre_build(custom_patches=patches)
