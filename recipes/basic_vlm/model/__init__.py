"""Model helpers for the basic_vlm recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen3_vl import (
        calculate_model_flops,
        patch_qwen3vl_conv3d,
        patch_qwen3vl_model_flops,
    )

__all__ = [
    "calculate_model_flops",
    "patch_qwen3vl_conv3d",
    "patch_qwen3vl_model_flops",
]


def __getattr__(name: str):
    """Lazily resolve Basic VLM model helper exports."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import qwen3_vl

    return getattr(qwen3_vl, name)
