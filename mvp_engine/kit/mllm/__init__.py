"""MLLM-specific kits."""

# ruff: noqa: F401

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import (
        MLLMBatchCollator,
        MLLMDataKit,
        MLLMDataSpec,
        MLLMDistributionSpec,
        MLLMLoaderSpec,
        MLLMMediaHandler,
        MLLMMediaSlot,
        MLLMMediaTypeHandler,
        MLLMPack,
        MLLMPackingAssembler,
        MLLMPackingSpec,
        MLLMSample,
        MLLMSampleSpec,
        MLLMSchemaHandler,
        MLLMSegment,
        MLLMSourceSpec,
        MLLMTextOnlyBatchGuard,
        MLLMTokenizationHandler,
        ModelInputs,
        QwenChatSchemaHandler,
        QwenImageHandler,
        QwenVLMediaHandler,
        QwenVLTokenizationHandler,
    )
    from .model import MLLMModelKit
    from .utils import Confidence, MLLMStepEstimationKit, StepEstimateResult

_EXPORT_MODULES = {
    "Confidence": ".utils",
    "MLLMBatchCollator": ".data",
    "MLLMDataKit": ".data",
    "MLLMDataSpec": ".data",
    "MLLMDistributionSpec": ".data",
    "MLLMLoaderSpec": ".data",
    "MLLMMediaHandler": ".data",
    "MLLMMediaSlot": ".data",
    "MLLMMediaTypeHandler": ".data",
    "MLLMModelKit": ".model",
    "MLLMPack": ".data",
    "MLLMPackingAssembler": ".data",
    "MLLMPackingSpec": ".data",
    "MLLMSample": ".data",
    "MLLMSampleSpec": ".data",
    "MLLMSchemaHandler": ".data",
    "MLLMSegment": ".data",
    "MLLMSourceSpec": ".data",
    "MLLMStepEstimationKit": ".utils",
    "MLLMTextOnlyBatchGuard": ".data",
    "MLLMTokenizationHandler": ".data",
    "ModelInputs": ".data",
    "QwenChatSchemaHandler": ".data",
    "QwenImageHandler": ".data",
    "QwenVLMediaHandler": ".data",
    "QwenVLTokenizationHandler": ".data",
    "StepEstimateResult": ".utils",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    """Lazily resolve MLLM kit exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
