"""Engine exports for the qwen3_vl recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen3_vl_engine import Qwen3VLEngine

__all__ = ["Qwen3VLEngine"]


def __getattr__(name: str):
    """Lazily resolve Qwen3-VL engine exports."""
    if name != "Qwen3VLEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .qwen3_vl_engine import Qwen3VLEngine

    return Qwen3VLEngine
