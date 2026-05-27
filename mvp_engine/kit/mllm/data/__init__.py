"""MLLM data-kit exports and collator factory."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import MLLMDataKit
    from .packing import (
        PackingAssembler,
        PackingOptions,
        build_packed_block_causal_mask,
        finalize_packed_samples,
    )
    from .process import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_PLACEHOLDER,
        IMAGE_TOKEN_PLACEHOLDER,
        MULTIMODAL_PLACEHOLDER,
        ROLE_MAP,
        THOUGHT_MARKERS,
        THOUGHT_PATTERN,
        THOUGHT_PREFIX,
        THOUGHT_SUFFIX,
        VISION_END_TOKEN,
        VISION_START_TOKEN,
        convert_images_to_pixel_values,
        process_sample,
        read_image,
    )
    from .types import ModelInputs


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
    "build_packed_block_causal_mask",
    "convert_images_to_pixel_values",
    "finalize_packed_samples",
    "read_image",
    "process_sample",
]

_EXPORT_MODULES = {
    "DEFAULT_IMAGE_TOKEN": ".process",
    "IMAGE_PLACEHOLDER": ".process",
    "IMAGE_TOKEN_PLACEHOLDER": ".process",
    "MLLMDataKit": ".data",
    "ModelInputs": ".types",
    "MULTIMODAL_PLACEHOLDER": ".process",
    "PackingAssembler": ".packing",
    "PackingOptions": ".packing",
    "ROLE_MAP": ".process",
    "THOUGHT_MARKERS": ".process",
    "THOUGHT_PATTERN": ".process",
    "THOUGHT_PREFIX": ".process",
    "THOUGHT_SUFFIX": ".process",
    "VISION_END_TOKEN": ".process",
    "VISION_START_TOKEN": ".process",
    "build_packed_block_causal_mask": ".packing",
    "convert_images_to_pixel_values": ".process",
    "finalize_packed_samples": ".packing",
    "process_sample": ".process",
    "read_image": ".process",
}


def __getattr__(name: str):
    """Lazily resolve MLLM data exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
