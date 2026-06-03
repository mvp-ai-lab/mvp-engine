"""Engine exports for the openbee recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .openbee_engine import OpenBeeEngine

__all__ = ["OpenBeeEngine"]


def __getattr__(name: str):
    """Lazily resolve OpenBee engine exports."""
    if name != "OpenBeeEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .openbee_engine import OpenBeeEngine

    return OpenBeeEngine
