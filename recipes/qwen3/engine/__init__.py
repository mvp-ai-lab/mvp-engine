"""Engine exports for the qwen3 recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen3_engine import Qwen3Engine

__all__ = ["Qwen3Engine"]


def __getattr__(name: str):
    """Lazily resolve the Qwen3 pretraining engine export."""
    if name != "Qwen3Engine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .qwen3_engine import Qwen3Engine

    return Qwen3Engine
