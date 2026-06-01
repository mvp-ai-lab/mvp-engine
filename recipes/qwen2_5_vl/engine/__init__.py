"""Engine exports for the qwen2_5_vl recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen2_5_vl_engine import Qwen2_5VLEngine

__all__ = ["Qwen2_5VLEngine"]


def __getattr__(name: str):
    """Lazily resolve engine exports."""
    if name != "Qwen2_5VLEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .qwen2_5_vl_engine import Qwen2_5VLEngine

    return Qwen2_5VLEngine

