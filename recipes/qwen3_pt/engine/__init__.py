"""Engine exports for the qwen3_pt recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen3_pt_engine import Qwen3PTEngine

__all__ = ["Qwen3PTEngine"]


def __getattr__(name: str):
    """Lazily resolve the Qwen3 pretraining engine export."""
    if name != "Qwen3PTEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .qwen3_pt_engine import Qwen3PTEngine

    return Qwen3PTEngine
