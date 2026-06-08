"""Reusable Liger Kernel integration helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

import torch

LigerStage = Literal["pre_build", "post_build"]
LigerModules = Literal["auto"] | dict[str, bool]
LigerModuleReplacer = Callable[[torch.nn.Module, Any], Iterable["LigerReplacement | dict[str, str]"]]

SUPPORTED_MODULES = frozenset(
    {
        "rms_norm",
        "layer_norm",
        "rope",
        "swiglu",
        "geglu",
        "cross_entropy",
        "fused_linear_cross_entropy",
    }
)
LOSS_MODULES = frozenset({"cross_entropy", "fused_linear_cross_entropy"})
PRE_BUILD_HELPERS = {
    "llama": "apply_liger_kernel_to_llama",
    "mistral": "apply_liger_kernel_to_mistral",
    "mixtral": "apply_liger_kernel_to_mixtral",
    "qwen2": "apply_liger_kernel_to_qwen2",
    "qwen2_vl": "apply_liger_kernel_to_qwen2_vl",
    "qwen3": "apply_liger_kernel_to_qwen3",
    "qwen3_vl": "apply_liger_kernel_to_qwen3_vl",
}
MODEL_FAMILY_UNSUPPORTED_MODULES = {
    # The supported Liger release accepts the kwarg but does not replace dense
    # Qwen3-VL MLP modules, so enabled SwiGLU would be a false-positive report.
    "qwen3_vl": frozenset({"swiglu"}),
}
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


@dataclass(frozen=True)
class LigerReplacement:
    """One post-build module replacement applied by LigerKernelKit."""

    path: str
    source: str
    target: str


@dataclass(frozen=True)
class LigerKernelReport:
    """Summary of one Liger Kernel application."""

    stage: LigerStage
    modules: dict[str, bool]
    model_family: str | None = None
    helper: str | None = None
    replacements: tuple[LigerReplacement, ...] = ()


class LigerKernelKit:
    """Apply Liger Kernel pre-build family patches or post-build replacements."""

    def resolve_modules(
        self,
        *,
        stage: LigerStage,
        modules: LigerModules = "auto",
        model_family: str | None = None,
        loss_kernels_allowed: bool = False,
    ) -> dict[str, bool]:
        """Resolve semantic Liger module selections for one stage."""
        family = _normalize_model_family(model_family)
        if modules == "auto":
            resolved = dict(PRE_BUILD_AUTO_MODULES if stage == "pre_build" else POST_BUILD_AUTO_MODULES)
            for module_name in MODEL_FAMILY_UNSUPPORTED_MODULES.get(family or "", frozenset()):
                resolved[module_name] = False
        else:
            unknown = sorted(set(modules) - SUPPORTED_MODULES)
            if unknown:
                raise ValueError(f"Unsupported Liger module(s): {unknown}.")
            resolved = {name: False for name in SUPPORTED_MODULES}
            resolved.update(modules)

        self._reject_unsupported_family_modules(resolved, model_family=family)
        if not loss_kernels_allowed:
            self._reject_loss_kernels(resolved)
        return resolved

    def apply_pre_build(
        self,
        *,
        model_family: str,
        modules: LigerModules = "auto",
        helper_name: str | None = None,
        loss_kernels_allowed: bool = False,
        strict: bool = True,
    ) -> LigerKernelReport:
        """Apply an official Liger model-family patch before model construction."""
        family = _normalize_model_family(model_family)
        if family is None:
            raise ValueError("model_family must be provided for pre-build Liger patching.")

        resolved_modules = self.resolve_modules(
            stage="pre_build",
            modules=modules,
            model_family=family,
            loss_kernels_allowed=loss_kernels_allowed,
        )
        helper = helper_name or PRE_BUILD_HELPERS.get(family)
        if helper is None:
            raise ValueError(f"No built-in Liger pre-build helper is registered for model family {model_family!r}.")

        liger_transformers = _import_liger_transformers()
        patch_fn = getattr(liger_transformers, helper, None)
        if patch_fn is None:
            raise RuntimeError(f"Liger Kernel does not expose `{helper}` in this environment.")

        patch_fn(**self._filter_patch_kwargs(patch_fn, resolved_modules, strict=strict))
        return LigerKernelReport(
            stage="pre_build",
            model_family=family,
            modules=resolved_modules,
            helper=helper,
        )

    def apply_post_build(
        self,
        model: torch.nn.Module,
        *,
        model_family: str | None = None,
        modules: LigerModules = "auto",
        module_replacers: dict[str, LigerModuleReplacer] | None = None,
        loss_kernels_allowed: bool = False,
        strict: bool = True,
    ) -> torch.nn.Module:
        """Apply Liger replacements to an already-built model instance."""
        resolved_modules = self.resolve_modules(
            stage="post_build",
            modules=modules,
            model_family=model_family,
            loss_kernels_allowed=loss_kernels_allowed,
        )
        liger_transformers = _import_liger_transformers()
        replacers = self._build_post_build_replacers(liger_transformers)
        if module_replacers:
            replacers.update(module_replacers)

        unsupported = sorted(name for name, enabled in resolved_modules.items() if enabled and name not in replacers)
        if unsupported and strict:
            raise ValueError(f"Liger post-build replacement does not support enabled module(s): {unsupported}.")

        replacements: list[LigerReplacement] = []
        for module_name, enabled in resolved_modules.items():
            if not enabled or module_name not in replacers:
                continue
            replacements.extend(_normalize_replacements(replacers[module_name](model, liger_transformers)))

        if not replacements and strict:
            raise RuntimeError("Liger post-build patch did not replace any modules.")

        model._mvp_engine_liger_kernel = LigerKernelReport(  # noqa: SLF001
            stage="post_build",
            model_family=_normalize_model_family(model_family),
            modules=resolved_modules,
            replacements=tuple(replacements),
        )
        return model

    def _reject_unsupported_family_modules(
        self,
        modules: dict[str, bool],
        *,
        model_family: str | None,
    ) -> None:
        unsupported_modules = MODEL_FAMILY_UNSUPPORTED_MODULES.get(model_family or "", frozenset())
        enabled_unsupported = sorted(name for name in unsupported_modules if modules.get(name, False))
        if enabled_unsupported:
            raise ValueError(
                f"Liger Kernel integration for model family {model_family!r} does not support enabled module(s): "
                f"{enabled_unsupported}."
            )

    def _reject_loss_kernels(self, modules: dict[str, bool]) -> None:
        enabled_loss_modules = sorted(name for name in LOSS_MODULES if modules.get(name, False))
        if enabled_loss_modules:
            raise ValueError(
                "Liger loss kernels are disabled by default because recipes may need custom loss accounting. "
                f"Enable an explicit compatibility path before using: {enabled_loss_modules}."
            )

    def _filter_patch_kwargs(
        self,
        patch_fn: Callable[..., Any],
        modules: dict[str, bool],
        *,
        strict: bool,
    ) -> dict[str, bool]:
        signature = inspect.signature(patch_fn)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return dict(modules)

        accepted_names = set(signature.parameters)
        unsupported = sorted(name for name, enabled in modules.items() if enabled and name not in accepted_names)
        if unsupported and strict:
            raise ValueError(f"Liger pre-build patch does not support enabled module(s): {unsupported}.")
        return {name: enabled for name, enabled in modules.items() if name in accepted_names}

    def _build_post_build_replacers(self, liger_transformers: Any) -> dict[str, LigerModuleReplacer]:
        return {
            "rms_norm": lambda model, _: _replace_rms_norm_modules(model, liger_transformers.LigerRMSNorm),
            "layer_norm": lambda model, _: _replace_layer_norm_modules(model, liger_transformers.LigerLayerNorm),
        }


def _import_liger_transformers() -> Any:
    try:
        import liger_kernel.transformers as liger_transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LigerKernelKit requires the optional `liger-kernel` package.") from exc
    return liger_transformers


def _replace_rms_norm_modules(
    model: torch.nn.Module, liger_rms_norm_cls: type[torch.nn.Module]
) -> list[LigerReplacement]:
    replacements: list[LigerReplacement] = []
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
            LigerReplacement(
                path=module_path,
                source=child.__class__.__name__,
                target=replacement.__class__.__name__,
            )
        )
    return replacements


def _replace_layer_norm_modules(
    model: torch.nn.Module,
    liger_layer_norm_cls: type[torch.nn.Module],
) -> list[LigerReplacement]:
    replacements: list[LigerReplacement] = []
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
            LigerReplacement(
                path=module_path,
                source=child.__class__.__name__,
                target=replacement.__class__.__name__,
            )
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


def _normalize_replacements(replacements: Iterable[LigerReplacement | dict[str, str]]) -> list[LigerReplacement]:
    normalized: list[LigerReplacement] = []
    for replacement in replacements:
        if isinstance(replacement, LigerReplacement):
            normalized.append(replacement)
            continue
        normalized.append(
            LigerReplacement(
                path=replacement["path"],
                source=replacement["source"],
                target=replacement["target"],
            )
        )
    return normalized


def _normalize_model_family(model_family: str | None) -> str | None:
    if model_family is None:
        return None
    normalized = model_family.strip().lower().replace("-", "_")
    return normalized or None
