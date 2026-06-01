"""Model helper exports for the qwen2_5_vl recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen2_5_vl import (
        calculate_model_flops,
        patch_qwen2_5vl_conv3d,
        patch_qwen2_5vl_model_flops,
    )

__all__ = [
    "calculate_model_flops",
    "patch_qwen2_5vl_conv3d",
    "patch_qwen2_5vl_model_flops",
]


def __getattr__(name: str):
    """Lazily resolve model helper exports."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import qwen2_5_vl

    return getattr(qwen2_5_vl, name)

