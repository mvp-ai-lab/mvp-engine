"""Reusable Liger Kernel integration, applied before model construction.

Liger ships one ``apply_liger_kernel_to_<family>`` per model, but every helper is
the same skeleton: a few flag-gated ``setattr(modeling_module, symbol, liger_impl)``
assignments. This kit keeps that single skeleton as one entry point, called
before the model is built:

* official families dispatch to liger's own helper;
* custom models (no official helper) pass an explicit ``{module: LigerPatch}`` map
  describing the same symbol swaps for their own modeling module.

Everything happens before the model is instantiated, so there is no module-tree
walking or instance fix-up. Loss kernels stay disabled unless explicitly allowed,
because recipes often own loss reduction and token normalization.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Literal

SUPPORTED_MODULES = frozenset(
    {"rope", "rms_norm", "layer_norm", "swiglu", "geglu", "cross_entropy", "fused_linear_cross_entropy"}
)
LOSS_MODULES = frozenset({"cross_entropy", "fused_linear_cross_entropy"})

# model_type values that reuse another family's liger helper
MODEL_TYPE_ALIASES = {"qwq": "qwen2", "qwen2_5": "qwen2", "qvq": "qwen2_vl"}


@dataclass(frozen=True)
class LigerPatch:
    """One symbol swap for a custom model: ``setattr(import_module(module), attr, replacement)``."""

    module: str
    attr: str
    replacement: Any


@dataclass(frozen=True)
class LigerKernelReport:
    """Summary of one Liger Kernel application.

    ``applied`` lists only the kwargs the kit forwarded (on the official auto route
    that is just the loss overrides), not liger's own per-model defaults.
    """

    model_family: str | None
    route: Literal["official", "custom"]
    helper: str | None = None
    applied: dict[str, bool] | None = None
    patched: tuple[str, ...] = ()


class LigerKernelKit:
    """Apply Liger Kernel before model construction via module-level monkey-patching."""

    def apply(
        self,
        model_name_or_path: str | None = None,
        *,
        model_family: str | None = None,
        modules: str | dict[str, bool] = "auto",
        custom_patches: dict[str, LigerPatch] | None = None,
        loss_kernels_allowed: bool = False,
        trust_remote_code: bool = True,
    ) -> LigerKernelReport:
        """Patch Liger kernels before the model is built.

        Without ``custom_patches``, dispatch to liger's official
        ``apply_liger_kernel_to_<family>`` (family inferred from the HF config when
        not given). Otherwise apply the given symbol swaps to the custom model's own
        modeling module. ``model_name_or_path`` is only used on the official route.
        Loss kernels require ``loss_kernels_allowed=True``.
        """
        if isinstance(modules, str) and modules != "auto":
            raise ValueError('`modules` must be "auto" or a dict[str, bool].')

        family = _normalize_family(model_family)
        if custom_patches is not None:
            return self._apply_custom(family, modules, custom_patches, loss_kernels_allowed)

        if family is None:
            if model_name_or_path is None:
                raise ValueError("Provide model_name_or_path or model_family for the official route.")
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
            family = _normalize_family(getattr(config, "model_type", None))
            if family is None:
                raise ValueError(f"Could not infer model family from config at {model_name_or_path!r}.")
        return self._apply_official(family, modules, loss_kernels_allowed)

    def _apply_official(self, family: str, modules: str | dict[str, bool], loss_allowed: bool) -> LigerKernelReport:
        try:
            import liger_kernel.transformers as liger_transformers
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("LigerKernelKit requires the optional `liger-kernel` package.") from exc

        candidates = dict.fromkeys(
            f"apply_liger_kernel_to_{name}" for name in (family, MODEL_TYPE_ALIASES.get(family)) if name
        )
        for helper in candidates:
            patch_fn = getattr(liger_transformers, helper, None)
            if patch_fn is not None:
                break
        else:
            raise RuntimeError(
                f"Liger Kernel exposes no official helper for model family {family!r} (tried: {', '.join(candidates)})."
            )

        accepted = set(inspect.signature(patch_fn).parameters)
        if modules == "auto":
            kwargs: dict[str, bool] = {}  # let liger pick per-model defaults (swiglu vs geglu, mRoPE, ...)
        else:
            _check_module_names(modules)
            unsupported = sorted(m for m, on in modules.items() if on and m not in accepted)
            if unsupported:
                raise ValueError(f"{helper} does not support enabled module(s): {unsupported}.")
            kwargs = dict(modules)

        if not loss_allowed:
            requested = sorted(m for m in LOSS_MODULES if kwargs.get(m))
            if requested:
                raise ValueError(f"Loss kernels need loss_kernels_allowed=True: {requested}.")
            kwargs.update({m: False for m in LOSS_MODULES})  # override liger's fused-CE-on-by-default

        kwargs = {m: on for m, on in kwargs.items() if m in accepted}
        patch_fn(**kwargs)
        return LigerKernelReport(model_family=family, route="official", helper=helper, applied=kwargs)

    def _apply_custom(
        self,
        family: str | None,
        modules: str | dict[str, bool],
        patches: dict[str, LigerPatch],
        loss_allowed: bool,
    ) -> LigerKernelReport:
        _check_module_names(patches)
        if modules == "auto":
            enabled = sorted(patches)
        else:
            _check_module_names(modules)
            enabled = sorted(m for m, on in modules.items() if on)
            missing = sorted(set(enabled) - set(patches))
            if missing:
                raise ValueError(f"No custom_patches provided for enabled module(s): {missing}.")

        blocked = sorted(set(enabled) & LOSS_MODULES)
        if blocked and not loss_allowed:
            raise ValueError(f"Loss kernels need loss_kernels_allowed=True: {blocked}.")

        patched = []
        for name in enabled:
            patch = patches[name]
            module = importlib.import_module(patch.module)
            if not hasattr(module, patch.attr):
                raise AttributeError(f"{patch.module!r} has no symbol {patch.attr!r} to patch (renamed upstream?).")
            setattr(module, patch.attr, patch.replacement)
            patched.append(f"{patch.module}.{patch.attr}")
        return LigerKernelReport(
            model_family=family,
            route="custom",
            applied=dict.fromkeys(enabled, True),
            patched=tuple(patched),
        )


def _normalize_family(model_family: str | None) -> str | None:
    if model_family is None:
        return None
    normalized = model_family.strip().lower().replace("-", "_").replace(".", "_")
    return normalized or None


def _check_module_names(names: dict[str, Any]) -> None:
    unknown = sorted(set(names) - SUPPORTED_MODULES)
    if unknown:
        raise ValueError(f"Unsupported Liger module(s): {unknown}. Choose from {sorted(SUPPORTED_MODULES)}.")
