"""Dataset helpers for the ViT image classification recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .imagenet import build_dataset, build_transforms

__all__ = ["build_dataset", "build_transforms"]


def __getattr__(name: str):
    """Lazily resolve ViT classification dataset exports."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import imagenet

    return getattr(imagenet, name)
