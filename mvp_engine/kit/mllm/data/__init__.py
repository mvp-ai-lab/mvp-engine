"""MLLM data-kit exports and collator factory."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import (
        MULTIMODAL_PLACEHOLDER,
        THOUGHT_MARKERS,
        THOUGHT_PATTERN,
        THOUGHT_PREFIX,
        THOUGHT_SUFFIX,
        MLLMDataKit,
    )
    from .media import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_PLACEHOLDER,
        VISION_END_TOKEN,
        VISION_START_TOKEN,
        MLLMMediaKit,
        read_image,
    )
    from .packing import (
        PackingAssembler,
        PackingOptions,
        build_packed_block_causal_mask,
        finalize_packed_samples,
    )
    from .sample import IMAGE_PLACEHOLDER, ROLE_MAP, MLLMSampleKit
    from .types import CanonicalMedia, CanonicalMLLMSample, ModelInputs
    from .video import VideoMediaKit


def MLLMCollator(
    *,
    pad_token_id: int,
    processor: Any,
    ignore_index: int = -100,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build the standard MLLM collator."""

    from .data import MLLMDataKit

    return MLLMDataKit().build_collator(
        pad_token_id=pad_token_id,
        processor=processor,
        ignore_index=ignore_index,
    )


__all__ = [
    "DEFAULT_IMAGE_TOKEN",
    "IMAGE_PLACEHOLDER",
    "IMAGE_TOKEN_PLACEHOLDER",
    "MLLMCollator",
    "MLLMDataKit",
    "MLLMMediaKit",
    "MLLMSampleKit",
    "ModelInputs",
    "MULTIMODAL_PLACEHOLDER",
    "PackingAssembler",
    "PackingOptions",
    "ROLE_MAP",
    "THOUGHT_MARKERS",
    "THOUGHT_PATTERN",
    "THOUGHT_PREFIX",
    "THOUGHT_SUFFIX",
    "VISION_END_TOKEN",
    "VISION_START_TOKEN",
    "VideoMediaKit",
    "build_packed_block_causal_mask",
    "CanonicalMedia",
    "CanonicalMLLMSample",
    "finalize_packed_samples",
    "read_image",
]

_EXPORT_MODULES = {
    "DEFAULT_IMAGE_TOKEN": ".media",
    "IMAGE_PLACEHOLDER": ".sample",
    "IMAGE_TOKEN_PLACEHOLDER": ".media",
    "MLLMDataKit": ".data",
    "MLLMMediaKit": ".media",
    "MLLMSampleKit": ".sample",
    "ModelInputs": ".types",
    "MULTIMODAL_PLACEHOLDER": ".data",
    "PackingAssembler": ".packing",
    "PackingOptions": ".packing",
    "ROLE_MAP": ".sample",
    "THOUGHT_MARKERS": ".data",
    "THOUGHT_PATTERN": ".data",
    "THOUGHT_PREFIX": ".data",
    "THOUGHT_SUFFIX": ".data",
    "VISION_END_TOKEN": ".media",
    "VISION_START_TOKEN": ".media",
    "VideoMediaKit": ".video",
    "build_packed_block_causal_mask": ".packing",
    "CanonicalMedia": ".types",
    "CanonicalMLLMSample": ".types",
    "finalize_packed_samples": ".packing",
    "read_image": ".media",
}


def __getattr__(name: str):
    """Lazily resolve MLLM data exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
