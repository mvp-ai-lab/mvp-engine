"""Model helpers for the Qwen3 pretraining recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen3 import calculate_model_flops, patch_qwen3_model_flops

__all__ = [
    "calculate_model_flops",
    "patch_qwen3_model_flops",
]


def __getattr__(name: str):
    """Lazily resolve Qwen3 model helper exports."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import qwen3

    return getattr(qwen3, name)
