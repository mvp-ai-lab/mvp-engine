"""Dataset adapters for the interleaved recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interleaved import (
        InterleavedDataGuard,
        InterleavedDataKit,
        InterleavedMediaKit,
        InterleavedSampleKit,
        build_interleaved_data_kit,
        infer_image_size,
    )

__all__ = [
    "InterleavedDataGuard",
    "InterleavedDataKit",
    "InterleavedMediaKit",
    "InterleavedSampleKit",
    "build_interleaved_data_kit",
    "infer_image_size",
]


def __getattr__(name: str):
    """Lazily resolve interleaved dataset helpers."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import interleaved

    return getattr(interleaved, name)
