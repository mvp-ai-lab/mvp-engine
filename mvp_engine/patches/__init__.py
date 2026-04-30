"""Runtime patches applied by MVP Engine before training starts."""

from .registry import PatchResult, apply_all_patches

__all__ = [
    "PatchResult",
    "apply_all_patches",
]
