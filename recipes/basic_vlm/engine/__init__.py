"""Engine exports for the basic_vlm recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basic_vlm_engine import BasicVLMEngine

__all__ = ["BasicVLMEngine"]


def __getattr__(name: str):
    """Lazily resolve Basic VLM engine exports."""
    if name != "BasicVLMEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .basic_vlm_engine import BasicVLMEngine

    return BasicVLMEngine
