"""Reusable text-only LM training kits."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import (
        LLMBatchCollator,
        LLMDataKit,
        LLMDataSpec,
        LLMDistributionSpec,
        LLMLoaderSpec,
        LLMPack,
        LLMPackingAssembler,
        LLMPackingSpec,
        LLMPretrainTextSchemaHandler,
        LLMPretrainTextTokenizationHandler,
        LLMSample,
        LLMSampleSpec,
        LLMSchemaHandler,
        LLMSegment,
        LLMSourceSpec,
        LLMTokenizationHandler,
        ModelInputs,
        QwenChatSchemaHandler,
        QwenChatTokenizationHandler,
        build_packed_block_causal_mask,
    )
    from .model import LLMModelKit
    from .utils import LLMStepEstimateResult, LLMStepEstimationKit

__all__ = [
    "LLMBatchCollator",
    "LLMDataSpec",
    "LLMDistributionSpec",
    "LLMDataKit",
    "LLMModelKit",
    "LLMLoaderSpec",
    "LLMPack",
    "LLMPackingAssembler",
    "LLMPackingSpec",
    "LLMPretrainTextSchemaHandler",
    "LLMPretrainTextTokenizationHandler",
    "LLMSample",
    "LLMSampleSpec",
    "LLMSchemaHandler",
    "LLMSegment",
    "LLMSourceSpec",
    "LLMStepEstimateResult",
    "LLMStepEstimationKit",
    "LLMTokenizationHandler",
    "ModelInputs",
    "QwenChatSchemaHandler",
    "QwenChatTokenizationHandler",
    "build_packed_block_causal_mask",
]

_EXPORT_MODULES = {
    "LLMBatchCollator": ".data",
    "LLMDataSpec": ".data",
    "LLMDistributionSpec": ".data",
    "LLMDataKit": ".data",
    "LLMLoaderSpec": ".data",
    "LLMPack": ".data",
    "LLMPackingAssembler": ".data",
    "LLMPackingSpec": ".data",
    "LLMPretrainTextSchemaHandler": ".data",
    "LLMPretrainTextTokenizationHandler": ".data",
    "LLMSample": ".data",
    "LLMSampleSpec": ".data",
    "LLMSchemaHandler": ".data",
    "LLMSegment": ".data",
    "LLMSourceSpec": ".data",
    "LLMStepEstimateResult": ".utils",
    "LLMStepEstimationKit": ".utils",
    "LLMTokenizationHandler": ".data",
    "ModelInputs": ".data",
    "QwenChatSchemaHandler": ".data",
    "QwenChatTokenizationHandler": ".data",
    "build_packed_block_causal_mask": ".data",
    "LLMModelKit": ".model",
}


def __getattr__(name: str):
    """Lazily resolve text-LM kit exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(_EXPORT_MODULES[name], __name__), name)
