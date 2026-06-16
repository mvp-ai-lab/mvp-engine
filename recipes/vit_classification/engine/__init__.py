"""Engine exports for the ViT image classification recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vit_classification_engine import ViTClassificationEngine

__all__ = ["ViTClassificationEngine"]


def __getattr__(name: str):
    """Lazily resolve ViT classification engine exports."""
    if name != "ViTClassificationEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .vit_classification_engine import ViTClassificationEngine

    return ViTClassificationEngine
