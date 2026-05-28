"""Engine exports for the minimal-vlm recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .minimal_vlm_engine import MinimalVLMEngine

__all__ = ["MinimalVLMEngine"]


def __getattr__(name: str):
    """Lazily resolve Minimal VLM engine exports."""
    if name != "MinimalVLMEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .minimal_vlm_engine import MinimalVLMEngine

    return MinimalVLMEngine
