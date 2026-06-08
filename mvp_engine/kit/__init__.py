"""Reusable training kits."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loss.token_loss import (
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
    from .model import LigerKernelKit, LigerKernelReport, LigerReplacement
    from .optim import OptimKit
    from .perf.mfu import MFUKit

__all__ = [
    "MFUKit",
    "LigerKernelKit",
    "LigerKernelReport",
    "LigerReplacement",
    "MLLMCollator",
    "MLLMDataKit",
    "MLLMMediaKit",
    "MLLMModelKit",
    "MLLMSampleKit",
    "ModelInputs",
    "OptimKit",
    "PackingOptions",
    "TokenNormedLossKit",
    "TokenLossStats",
    "apply_chunked_token_loss_patch",
]

_EXPORT_MODULES = {
    "MFUKit": ".perf.mfu",
    "LigerKernelKit": ".model",
    "LigerKernelReport": ".model",
    "LigerReplacement": ".model",
    "MLLMCollator": ".mllm",
    "MLLMDataKit": ".mllm",
    "MLLMMediaKit": ".mllm",
    "MLLMModelKit": ".mllm",
    "MLLMSampleKit": ".mllm",
    "ModelInputs": ".mllm",
    "OptimKit": ".optim",
    "PackingOptions": ".mllm",
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
