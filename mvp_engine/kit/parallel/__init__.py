"""Parallel-layout helpers for reusable training kits."""

from .cp import CPKit, CPSequenceSpec, QwenVLCPKit

__all__ = [
    "CPKit",
    "CPSequenceSpec",
    "QwenVLCPKit",
]
