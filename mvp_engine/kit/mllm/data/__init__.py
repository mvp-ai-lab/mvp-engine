"""MLLM data-kit exports."""

# ruff: noqa: F401

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .collator import MLLMBatchCollator
    from .data import MLLMDataKit
    from .guard import MLLMTextOnlyBatchGuard
    from .media import MLLMMediaHandler, MLLMMediaTypeHandler
    from .packing import MLLMPackingAssembler, build_packed_block_causal_mask
    from .qwen import (
        QwenImageHandler,
        QwenVLChatSchemaHandler,
        QwenVLMediaHandler,
        QwenVLTokenizationHandler,
    )
    from .sample import MLLMPack, MLLMSample
    from .schema import MLLMSchemaHandler
    from .spec import (
        MLLMDataSpec,
        MLLMDistributionSpec,
        MLLMLoaderSpec,
        MLLMPackingSpec,
        MLLMSampleSpec,
        MLLMSourceSpec,
    )
    from .tokenization import MLLMTokenizationHandler
    from .types import MLLMMediaSlot, MLLMSegment, ModelInputs


_EXPORT_MODULES = {
    "MLLMBatchCollator": ".collator",
    "MLLMDataKit": ".data",
    "MLLMDataSpec": ".spec",
    "MLLMDistributionSpec": ".spec",
    "MLLMLoaderSpec": ".spec",
    "MLLMMediaHandler": ".media",
    "MLLMMediaSlot": ".types",
    "MLLMMediaTypeHandler": ".media",
    "MLLMPack": ".sample",
    "MLLMPackingAssembler": ".packing",
    "MLLMPackingSpec": ".spec",
    "MLLMSample": ".sample",
    "MLLMSampleSpec": ".spec",
    "MLLMSchemaHandler": ".schema",
    "MLLMSegment": ".types",
    "MLLMSourceSpec": ".spec",
    "MLLMTextOnlyBatchGuard": ".guard",
    "MLLMTokenizationHandler": ".tokenization",
    "ModelInputs": ".types",
    "QwenImageHandler": ".qwen",
    "QwenVLChatSchemaHandler": ".qwen",
    "QwenVLMediaHandler": ".qwen",
    "QwenVLTokenizationHandler": ".qwen",
    "build_packed_block_causal_mask": ".packing",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    """Lazily resolve MLLM data exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
