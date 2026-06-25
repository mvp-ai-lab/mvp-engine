"""Packed-input helper exports for the qwen2_5_vl recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen2_5_vl import (
        build_packed_fa2_varlen_kwargs,
        build_qwen2_5_vl_packed_position_ids,
        prepare_packed_model_inputs,
    )

__all__ = [
    "build_packed_fa2_varlen_kwargs",
    "build_qwen2_5_vl_packed_position_ids",
    "prepare_packed_model_inputs",
]


def __getattr__(name: str):
    """Lazily resolve packed-input helper exports."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import qwen2_5_vl

    return getattr(qwen2_5_vl, name)
