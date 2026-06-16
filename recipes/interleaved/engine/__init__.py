"""Engine exports for the interleaved recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interleaved_engine import InterleavedEngine

__all__ = ["InterleavedEngine"]


def __getattr__(name: str):
    """Lazily resolve interleaved engine exports."""
    if name != "InterleavedEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .interleaved_engine import InterleavedEngine

    return InterleavedEngine
