"""Packed-attention helpers for the Qwen3 pretraining recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prepare import build_packed_text_position_ids, prepare_packed_model_inputs

__all__ = [
    "build_packed_text_position_ids",
    "prepare_packed_model_inputs",
]


def __getattr__(name: str):
    """Lazily resolve Qwen3 packing helper exports."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import prepare

    return getattr(prepare, name)
