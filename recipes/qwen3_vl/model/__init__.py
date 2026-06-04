"""Model helpers for the qwen3_vl recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .liger import (
        apply_liger_kernel_pre_build,
        patch_liger_kernel_post_build,
    )
    from .qwen3_vl import (
        calculate_model_flops,
        patch_qwen3vl_conv3d,
        patch_qwen3vl_model_flops,
    )

__all__ = [
    "apply_liger_kernel_pre_build",
    "calculate_model_flops",
    "patch_liger_kernel_post_build",
    "patch_qwen3vl_conv3d",
    "patch_qwen3vl_model_flops",
]


def __getattr__(name: str):
    """Lazily resolve Qwen3-VL model helper exports."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in {"apply_liger_kernel_pre_build", "patch_liger_kernel_post_build"}:
        from . import liger

        return getattr(liger, name)

    from . import qwen3_vl

    return getattr(qwen3_vl, name)
