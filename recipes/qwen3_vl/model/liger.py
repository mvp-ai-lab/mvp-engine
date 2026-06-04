"""Qwen3-VL recipe-local Liger Kernel integration."""

from __future__ import annotations

import inspect
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..configs.schema import Qwen3VLLigerKernelConfig

SUPPORTED_MODULES = {
    "rms_norm",
    "layer_norm",
    "rope",
    "swiglu",
    "geglu",
    "cross_entropy",
    "fused_linear_cross_entropy",
}
LOSS_MODULES = {"cross_entropy", "fused_linear_cross_entropy"}
PRE_BUILD_AUTO_MODULES = {
    "rms_norm": True,
    "rope": True,
    "swiglu": True,
    "layer_norm": False,
    "geglu": False,
    "cross_entropy": False,
    "fused_linear_cross_entropy": False,
}
POST_BUILD_AUTO_MODULES = {
    "rms_norm": True,
    "layer_norm": False,
    "rope": False,
    "swiglu": False,
    "geglu": False,
    "cross_entropy": False,
    "fused_linear_cross_entropy": False,
}
POST_BUILD_REPLACERS = {"rms_norm", "layer_norm"}


def apply_liger_kernel_pre_build(*, model_name_or_path: str, config: Qwen3VLLigerKernelConfig) -> None:
    """Apply official Qwen3-VL Liger monkey patches before model construction."""
    modules, explicit_modules = _resolve_modules(config, stage="pre_build")
    _reject_loss_kernels(modules)

    try:
        import liger_kernel.transformers as liger_transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`model.liger_kernel.enabled=true` requires the optional `liger-kernel` package."
        ) from exc

    patch_fn = getattr(liger_transformers, "apply_liger_kernel_to_qwen3_vl", None)
    if patch_fn is None:
        raise RuntimeError(
            "Liger Kernel does not expose `apply_liger_kernel_to_qwen3_vl` in this environment. "
            f"Cannot pre-build patch Qwen3-VL model {model_name_or_path!r}."
        )

    patch_fn(**_filter_patch_kwargs(patch_fn, modules, explicit_modules=explicit_modules))


def patch_liger_kernel_post_build(
    model: torch.nn.Module,
    *,
    config: Qwen3VLLigerKernelConfig,
) -> torch.nn.Module:
    """Replace supported Qwen3-VL modules on an already-built model instance."""
    modules, explicit_modules = _resolve_modules(config, stage="post_build")
    _reject_loss_kernels(modules)
    _reject_unsupported_post_build_modules(modules, explicit_modules=explicit_modules)

    try:
        import liger_kernel.transformers as liger_transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`model.liger_kernel.enabled=true` requires the optional `liger-kernel` package."
        ) from exc

    replacements: list[dict[str, str]] = []
    if modules.get("rms_norm", False):
        replacements.extend(_replace_rms_norm_modules(model, liger_transformers.LigerRMSNorm))
    if modules.get("layer_norm", False):
        replacements.extend(_replace_layer_norm_modules(model, liger_transformers.LigerLayerNorm))

    if not replacements:
        raise RuntimeError("Qwen3-VL post-build Liger patch did not replace any modules.")

    model._mvp_engine_liger_kernel = {  # noqa: SLF001
        "stage": "post_build",
        "modules": modules,
        "replacements": replacements,
    }
    return model


def _resolve_modules(
    config: Qwen3VLLigerKernelConfig,
    *,
    stage: str,
) -> tuple[dict[str, bool], bool]:
    if config.modules == "auto":
        return dict(PRE_BUILD_AUTO_MODULES if stage == "pre_build" else POST_BUILD_AUTO_MODULES), False

    unknown = sorted(set(config.modules) - SUPPORTED_MODULES)
    if unknown:
        raise ValueError(f"Unsupported Qwen3-VL Liger module(s): {unknown}.")

    modules = {name: False for name in SUPPORTED_MODULES}
    modules.update(config.modules)
    return modules, True


def _reject_loss_kernels(modules: dict[str, bool]) -> None:
    enabled_loss_modules = sorted(name for name in LOSS_MODULES if modules.get(name, False))
    if enabled_loss_modules:
        raise ValueError(
            "Qwen3-VL uses TokenNormedLossKit.apply_chunked_token_loss_patch(...), so Liger loss kernels "
            f"are disabled for now: {enabled_loss_modules}."
        )


def _reject_unsupported_post_build_modules(modules: dict[str, bool], *, explicit_modules: bool) -> None:
    unsupported = sorted(name for name, enabled in modules.items() if enabled and name not in POST_BUILD_REPLACERS)
    if unsupported and explicit_modules:
        raise ValueError(f"Qwen3-VL post-build Liger replacement does not support module(s): {unsupported}.")


def _filter_patch_kwargs(
    patch_fn: Any,
    modules: dict[str, bool],
    *,
    explicit_modules: bool,
) -> dict[str, bool]:
    signature = inspect.signature(patch_fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {name: enabled for name, enabled in modules.items() if enabled or explicit_modules}

    accepted_names = set(signature.parameters)
    unsupported = sorted(name for name, enabled in modules.items() if enabled and name not in accepted_names)
    if explicit_modules and unsupported:
        raise ValueError(f"Liger pre-build patch does not support explicitly requested module(s): {unsupported}.")
    # Forward every signature-accepted module with its real boolean, including the
    # disabled ones. Passing False explicitly overrides Liger's own kwarg defaults
    # (e.g. `fused_linear_cross_entropy=True` in 0.8.0), so a kernel the recipe left
    # off cannot be silently re-enabled behind _reject_loss_kernels.
    return {name: enabled for name, enabled in modules.items() if name in accepted_names}


def _replace_rms_norm_modules(model: torch.nn.Module, liger_rms_norm_cls: type[torch.nn.Module]) -> list[dict[str, str]]:
    replacements: list[dict[str, str]] = []
    for module_path, parent, child_name, child in _iter_replaceable_children(model):
        if _is_liger_module(child) or "rmsnorm" not in child.__class__.__name__.lower():
            continue
        if not hasattr(child, "weight"):
            continue

        replacement = _build_liger_norm(
            liger_rms_norm_cls,
            source=child,
            eps=_get_norm_eps(child),
            with_bias=False,
        )
        parent.add_module(child_name, replacement)
        replacements.append(
            {
                "path": module_path,
                "source": child.__class__.__name__,
                "target": replacement.__class__.__name__,
            }
        )
    return replacements


def _replace_layer_norm_modules(
    model: torch.nn.Module,
    liger_layer_norm_cls: type[torch.nn.Module],
) -> list[dict[str, str]]:
    replacements: list[dict[str, str]] = []
    for module_path, parent, child_name, child in _iter_replaceable_children(model):
        if _is_liger_module(child) or not isinstance(child, torch.nn.LayerNorm):
            continue

        replacement = _build_liger_norm(
            liger_layer_norm_cls,
            source=child,
            eps=float(child.eps),
            with_bias=child.bias is not None,
        )
        parent.add_module(child_name, replacement)
        replacements.append(
            {
                "path": module_path,
                "source": child.__class__.__name__,
                "target": replacement.__class__.__name__,
            }
        )
    return replacements


def _iter_replaceable_children(model: torch.nn.Module):
    for parent_path, parent in model.named_modules():
        for child_name, child in parent.named_children():
            module_path = child_name if not parent_path else f"{parent_path}.{child_name}"
            yield module_path, parent, child_name, child


def _build_liger_norm(
    liger_norm_cls: type[torch.nn.Module],
    *,
    source: torch.nn.Module,
    eps: float,
    with_bias: bool,
) -> torch.nn.Module:
    weight = getattr(source, "weight")
    replacement = _instantiate_liger_norm(liger_norm_cls, int(weight.shape[0]), eps=eps, with_bias=with_bias)
    replacement = replacement.to(device=weight.device, dtype=weight.dtype)

    with torch.no_grad():
        replacement.weight.copy_(weight)
        if with_bias and hasattr(replacement, "bias") and getattr(source, "bias", None) is not None:
            replacement.bias.copy_(source.bias)
    replacement.weight.requires_grad = weight.requires_grad
    if with_bias and hasattr(replacement, "bias") and getattr(source, "bias", None) is not None:
        replacement.bias.requires_grad = source.bias.requires_grad
    return replacement


def _instantiate_liger_norm(
    liger_norm_cls: type[torch.nn.Module],
    hidden_size: int,
    *,
    eps: float,
    with_bias: bool,
) -> torch.nn.Module:
    signature = inspect.signature(liger_norm_cls)
    kwargs: dict[str, Any] = {}
    if "eps" in signature.parameters:
        kwargs["eps"] = eps
    if "bias" in signature.parameters:
        kwargs["bias"] = with_bias

    try:
        return liger_norm_cls(hidden_size, **kwargs)
    except TypeError:
        kwargs["hidden_size"] = hidden_size
        return liger_norm_cls(**kwargs)


def _get_norm_eps(module: torch.nn.Module) -> float:
    for attr_name in ("variance_epsilon", "eps", "epsilon"):
        if hasattr(module, attr_name):
            return float(getattr(module, attr_name))
    return 1e-6


def _is_liger_module(module: torch.nn.Module) -> bool:
    return module.__class__.__module__.startswith("liger_kernel.")
