"""Engine exports for the video-mllm recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .video_mllm_engine import VideoMLLMEngine

__all__ = ["VideoMLLMEngine"]


def __getattr__(name: str):
    """Lazily resolve video MLLM engine exports."""
    if name != "VideoMLLMEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .video_mllm_engine import VideoMLLMEngine

    return VideoMLLMEngine
