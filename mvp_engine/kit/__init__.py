"""Reusable training kits."""

# ruff: noqa: F401

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
        Confidence,
        MLLMBatchCollator,
        MLLMDataKit,
        MLLMDataSpec,
        MLLMDistributionSpec,
        MLLMLoaderSpec,
        MLLMMediaHandler,
        MLLMMediaSlot,
        MLLMMediaTypeHandler,
        MLLMModelKit,
        MLLMPack,
        MLLMPackingAssembler,
        MLLMPackingSpec,
        MLLMSample,
        MLLMSampleSpec,
        MLLMSchemaHandler,
        MLLMSegment,
        MLLMSourceSpec,
        MLLMStepEstimationKit,
        MLLMTextOnlyBatchGuard,
        MLLMTokenizationHandler,
        ModelInputs,
        QwenChatSchemaHandler,
        QwenImageHandler,
        QwenVLMediaHandler,
        QwenVLTokenizationHandler,
        StepEstimateResult,
    )
    from .model import LigerKernelKit, LigerKernelReport, LigerPatch
    from .optim import OptimKit
    from .parallel import CPKit
    from .perf.mfu import MFUKit
    from .util import StepCountingKit, StepCountResult

_EXPORT_MODULES = {
    "Confidence": ".mllm",
    "LossGuard": ".loss.loss",
    "LossKit": ".loss.loss",
    "CPKit": ".parallel",
    "MFUKit": ".perf.mfu",
    "LigerKernelKit": ".model",
    "LigerKernelReport": ".model",
    "LigerPatch": ".model",
    "MLLMBatchCollator": ".mllm",
    "MLLMDataKit": ".mllm",
    "MLLMDataSpec": ".mllm",
    "MLLMDistributionSpec": ".mllm",
    "MLLMLoaderSpec": ".mllm",
    "MLLMMediaHandler": ".mllm",
    "MLLMMediaSlot": ".mllm",
    "MLLMMediaTypeHandler": ".mllm",
    "MLLMModelKit": ".mllm",
    "MLLMPack": ".mllm",
    "MLLMPackingAssembler": ".mllm",
    "MLLMPackingSpec": ".mllm",
    "MLLMSample": ".mllm",
    "MLLMSampleSpec": ".mllm",
    "MLLMSchemaHandler": ".mllm",
    "MLLMSegment": ".mllm",
    "MLLMSourceSpec": ".mllm",
    "MLLMStepEstimationKit": ".mllm",
    "MLLMTextOnlyBatchGuard": ".mllm",
    "MLLMTokenizationHandler": ".mllm",
    "ModelInputs": ".mllm",
    "OptimKit": ".optim",
    "PerTokenLossGuard": ".loss.token_loss",
    "QwenChatSchemaHandler": ".mllm",
    "QwenImageHandler": ".mllm",
    "QwenVLMediaHandler": ".mllm",
    "QwenVLTokenizationHandler": ".mllm",
    "StepCountResult": ".util",
    "StepCountingKit": ".util",
    "StepEstimateResult": ".mllm",
    "TokenLossStats": ".loss.token_loss",
    "TokenNormedLossKit": ".loss.token_loss",
    "apply_chunked_token_loss_patch": ".loss.token_loss",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    """Lazily resolve training-kit exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
