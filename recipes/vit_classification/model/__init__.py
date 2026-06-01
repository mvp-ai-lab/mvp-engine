"""Model helpers for the ViT image classification recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vit import build_vit_model

__all__ = ["build_vit_model"]


def __getattr__(name: str):
    """Lazily resolve ViT classification model exports."""
    if name != "build_vit_model":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .vit import build_vit_model

    return build_vit_model
