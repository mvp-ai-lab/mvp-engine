"""Reusable Liger Kernel integration, applied before model construction.

Liger ships one ``apply_liger_kernel_to_<family>`` per model, but every helper is
the same skeleton: a few flag-gated ``setattr(modeling_module, symbol, liger_impl)``
assignments. This kit keeps that single skeleton as one entry point, called
before the model is built:

* official families dispatch to liger's own helper;
* custom models (no official helper) pass an explicit ``{module: LigerPatch}`` map
  describing the same symbol swaps for their own modeling module.

Everything happens before the model is instantiated, so there is no module-tree
walking or instance fix-up. Loss kernels are excluded by default because recipes
often own loss reduction and token normalization; pass ``excluded_modules=()`` to
lift that.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Literal

# kwargs accepted across liger's official apply_liger_kernel_to_<family> helpers
SUPPORTED_MODULES = frozenset(
    {"rope", "rms_norm", "layer_norm", "swiglu", "geglu", "cross_entropy", "fused_linear_cross_entropy"}
)
# liger enables fused linear cross entropy by default; recipes often own loss accounting
DEFAULT_EXCLUDED_MODULES = ("cross_entropy", "fused_linear_cross_entropy")


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
    that is just the excluded-module overrides), not liger's own per-model defaults.
    """

    model_family: str | None
    route: Literal["official", "custom"]
    helper: str | None = None
    applied: dict[str, bool] | None = None
    patched: tuple[str, ...] = ()


class LigerKernelKit:
    """Apply Liger Kernel before model construction via module-level monkey-patching."""

    Patch = LigerPatch
    Report = LigerKernelReport

    def apply(
        self,
        model_name_or_path: str | None = None,
        *,
        model_family: str | None = None,
        modules: str | dict[str, bool] = "auto",
        excluded_modules: tuple[str, ...] = DEFAULT_EXCLUDED_MODULES,
        custom_patches: dict[str, LigerPatch] | None = None,
        trust_remote_code: bool = True,
    ) -> LigerKernelReport:
        """Patch Liger kernels before the model is built.

        Without ``custom_patches``, dispatch to liger's official
        ``apply_liger_kernel_to_<family>`` (family taken verbatim from ``model_family``
        or the HF config's ``model_type``). With ``custom_patches``, apply every given
        symbol swap to the custom model's own modeling module; ``modules`` and
        ``model_name_or_path`` only affect the official route. ``excluded_modules``
        disables the named kernels on either route and defaults to the loss kernels.
        """
        if isinstance(modules, str) and modules != "auto":
            raise ValueError('`modules` must be "auto" or a dict[str, bool].')

        excluded = set(excluded_modules)
        if custom_patches is not None:
            return self._apply_custom(model_family, custom_patches, excluded)

        family = model_family
        if family is None:
            if model_name_or_path is None:
                raise ValueError("Provide model_name_or_path or model_family for the official route.")
            from transformers import AutoConfig

            family = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code).model_type
        return self._apply_official(family, modules, excluded)

    def _apply_official(self, family: str, modules: str | dict[str, bool], excluded: set[str]) -> LigerKernelReport:
        import liger_kernel.transformers as liger_transformers  # lazy: optional dependency

        helper = f"apply_liger_kernel_to_{family}"
        patch_fn = getattr(liger_transformers, helper)  # unsupported family -> AttributeError naming the helper

        accepted = set(inspect.signature(patch_fn).parameters)
        if modules == "auto":
            kwargs: dict[str, bool] = {}  # let liger pick per-model defaults (swiglu vs geglu, mRoPE, ...)
        else:
            _check_module_names(modules)
            unsupported = sorted(m for m, on in modules.items() if on and m not in accepted)
            if unsupported:
                raise ValueError(f"{helper} does not support enabled module(s): {unsupported}.")
            kwargs = dict(modules)

        _check_module_names(excluded)
        kwargs.update({m: False for m in excluded})  # e.g. override liger's fused-CE-on-by-default

        kwargs = {m: on for m, on in kwargs.items() if m in accepted}
        patch_fn(**kwargs)
        return LigerKernelReport(model_family=family, route="official", helper=helper, applied=kwargs)

    def _apply_custom(
        self, family: str | None, patches: dict[str, LigerPatch], excluded: set[str]
    ) -> LigerKernelReport:
        enabled = sorted(set(patches) - excluded)

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


def _check_module_names(names) -> None:
    unknown = sorted(set(names) - SUPPORTED_MODULES)
    if unknown:
        raise ValueError(f"Unsupported Liger module(s): {unknown}. Choose from {sorted(SUPPORTED_MODULES)}.")
