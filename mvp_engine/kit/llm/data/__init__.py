"""LLM data-kit exports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .collator import LLMBatchCollator
    from .data import LLMDataKit
    from .guard import LLMModelInputGuard, LLMRawRowGuard, LLMSampleGuard
    from .packing import LLMPackingAssembler, build_packed_block_causal_mask
    from .qwen import QwenChatSchemaHandler, QwenChatTokenizationHandler
    from .sample import LLMPack, LLMSample
    from .schema import LLMPretrainTextSchemaHandler, LLMSchemaHandler
    from .spec import (
        LLMDataSpec,
        LLMDistributionSpec,
        LLMLoaderSpec,
        LLMPackingSpec,
        LLMSampleSpec,
        LLMSourceSpec,
    )
    from .tokenization import LLMPretrainTextTokenizationHandler, LLMTokenizationHandler
    from .types import LLMSegment, ModelInputs

__all__ = [
    "LLMBatchCollator",
    "LLMDataSpec",
    "LLMDistributionSpec",
    "LLMDataKit",
    "LLMLoaderSpec",
    "LLMModelInputGuard",
    "LLMPack",
    "LLMPackingAssembler",
    "LLMPackingSpec",
    "LLMPretrainTextSchemaHandler",
    "LLMPretrainTextTokenizationHandler",
    "QwenChatSchemaHandler",
    "QwenChatTokenizationHandler",
    "LLMRawRowGuard",
    "LLMSample",
    "LLMSampleGuard",
    "LLMSampleSpec",
    "LLMSchemaHandler",
    "LLMSegment",
    "LLMSourceSpec",
    "LLMTokenizationHandler",
    "ModelInputs",
    "build_packed_block_causal_mask",
]

_EXPORT_MODULES = {
    "LLMBatchCollator": ".collator",
    "LLMDataSpec": ".spec",
    "LLMDistributionSpec": ".spec",
    "LLMDataKit": ".data",
    "LLMLoaderSpec": ".spec",
    "LLMModelInputGuard": ".guard",
    "LLMPack": ".sample",
    "LLMPackingAssembler": ".packing",
    "LLMPackingSpec": ".spec",
    "LLMPretrainTextSchemaHandler": ".schema",
    "LLMPretrainTextTokenizationHandler": ".tokenization",
    "QwenChatSchemaHandler": ".qwen",
    "QwenChatTokenizationHandler": ".qwen",
    "LLMRawRowGuard": ".guard",
    "LLMSample": ".sample",
    "LLMSampleGuard": ".guard",
    "LLMSampleSpec": ".spec",
    "LLMSchemaHandler": ".schema",
    "LLMSegment": ".types",
    "LLMSourceSpec": ".spec",
    "LLMTokenizationHandler": ".tokenization",
    "ModelInputs": ".types",
    "build_packed_block_causal_mask": ".packing",
}


def __getattr__(name: str):
    """Lazily resolve LLM data exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(_EXPORT_MODULES[name], __name__), name)
