"""LLM data-kit exports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import LLMCollator, LLMDataKit, TokenizeAssembler
    from .guard import DataGuard
    from .packing import (
        PackingAssembler,
        PackingOptions,
        build_packed_block_causal_mask,
    )
    from .types import ModelInputs

# NOTE: `finalize_packed_samples` from `.packing` is intentionally NOT exported here.
# Use `LLMDataKit.finalize_packed_samples`, which also masks the loss at document
# boundaries; the raw `.packing` version would skip that masking.
__all__ = [
    "LLMCollator",
    "LLMDataKit",
    "TokenizeAssembler",
    "DataGuard",
    "ModelInputs",
    "PackingAssembler",
    "PackingOptions",
    "build_packed_block_causal_mask",
]

_EXPORT_MODULES = {
    "LLMCollator": ".data",
    "LLMDataKit": ".data",
    "TokenizeAssembler": ".data",
    "DataGuard": ".guard",
    "ModelInputs": ".types",
    "PackingAssembler": ".packing",
    "PackingOptions": ".packing",
    "build_packed_block_causal_mask": ".packing",
}


def __getattr__(name: str):
    """Lazily resolve LLM data exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(_EXPORT_MODULES[name], __name__), name)
