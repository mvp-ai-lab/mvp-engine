"""Reusable training kits."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loss.loss import LossGuard, LossKit
    from .loss.token_loss import (
        PerTokenLossGuard,
        TokenLossStats,
        TokenNormedLossKit,
        apply_chunked_token_loss_patch,
    )
    from .mllm import (
        MLLMCollator,
        MLLMDataKit,
        MLLMMediaKit,
        MLLMModelKit,
        MLLMSampleKit,
        ModelInputs,
        PackingOptions,
    )
    from .optim import OptimKit
    from .perf.mfu import MFUKit

__all__ = [
    "LossGuard",
    "LossKit",
    "MFUKit",
    "MLLMCollator",
    "MLLMDataKit",
    "MLLMMediaKit",
    "MLLMModelKit",
    "MLLMSampleKit",
    "ModelInputs",
    "OptimKit",
    "PackingOptions",
    "PerTokenLossGuard",
    "TokenNormedLossKit",
    "TokenLossStats",
    "apply_chunked_token_loss_patch",
]

_EXPORT_MODULES = {
    "LossGuard": ".loss.loss",
    "LossKit": ".loss.loss",
    "MFUKit": ".perf.mfu",
    "MLLMCollator": ".mllm",
    "MLLMDataKit": ".mllm",
    "MLLMMediaKit": ".mllm",
    "MLLMModelKit": ".mllm",
    "MLLMSampleKit": ".mllm",
    "ModelInputs": ".mllm",
    "OptimKit": ".optim",
    "PackingOptions": ".mllm",
    "PerTokenLossGuard": ".loss.token_loss",
    "TokenLossStats": ".loss.token_loss",
    "TokenNormedLossKit": ".loss.token_loss",
    "apply_chunked_token_loss_patch": ".loss.token_loss",
}


def __getattr__(name: str):
    """Lazily resolve training-kit exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
