"""Reusable text-only LM training kits."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import (
        LLMCollator,
        LLMDataKit,
        ModelInputs,
        PackingOptions,
        build_packed_block_causal_mask,
    )
    from .model import LLMModelKit

__all__ = [
    "LLMDataKit",
    "LLMModelKit",
    "LLMCollator",
    "ModelInputs",
    "PackingOptions",
    "build_packed_block_causal_mask",
]

_EXPORT_MODULES = {
    "LLMDataKit": ".data",
    "LLMCollator": ".data",
    "ModelInputs": ".data",
    "PackingOptions": ".data",
    "build_packed_block_causal_mask": ".data",
    "LLMModelKit": ".model",
}


def __getattr__(name: str):
    """Lazily resolve text-LM kit exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(_EXPORT_MODULES[name], __name__), name)
