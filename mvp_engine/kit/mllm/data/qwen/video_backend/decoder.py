"""Raw-video decoding for the Qwen video data path (PyAV backend).

Frame *selection* lives in ``sampling.py``; this module only fetches frames by
index. ``av`` (PyAV) is imported lazily and must be installed to decode video at
training time (``uv add av``); it is not a hard dependency of ``mvp_engine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VideoMeta:
    """Lightweight probe result describing one source video."""

    total_num_frames: int
    fps: float | None
    height: int
    width: int
    duration: float


def probe_video(path: str | Path) -> VideoMeta:
    """Read frame count, fps, and resolution without decoding all frames."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Leave fps as None (not 0.0) when unknown so the Qwen3-VL processor applies
        # its own fps fallback for timestamps instead of dividing by zero.
        fps = float(stream.average_rate) if stream.average_rate else None

        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration) / 1_000_000.0
        else:
            duration = 0.0

        total = int(stream.frames or 0)
        if total <= 0 and fps and duration:
            total = int(round(duration * fps))
        if total <= 0:
            raise ValueError(f"Could not determine frame count for video: {path}")

        return VideoMeta(
            total_num_frames=total,
            fps=fps,
            height=int(stream.height),
            width=int(stream.width),
            duration=duration,
        )


def decode_frames(path: str | Path, indices: list[int]) -> np.ndarray:
    """Decode the given frame indices and return an ``(T, H, W, 3)`` uint8 array.

    Indices are decoded in a single forward pass. If an index overruns the real
    stream length (e.g. an estimated frame count), the last decoded frame is
    repeated so the returned count always matches ``len(indices)``.
    """
    import av

    if not indices:
        raise ValueError("No frame indices requested for decoding.")

    targets = {int(i) for i in indices}
    last_index = max(targets)
    decoded: dict[int, np.ndarray] = {}

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for position, frame in enumerate(container.decode(stream)):
            if position in targets:
                decoded[position] = frame.to_ndarray(format="rgb24")
            if position >= last_index:
                break

    if not decoded:
        raise ValueError(f"Failed to decode any frame from video: {path}")

    fallback = decoded[max(decoded)]
    frames = [decoded.get(int(i), fallback) for i in indices]
    return np.stack(frames, axis=0)
