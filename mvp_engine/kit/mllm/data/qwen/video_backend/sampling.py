"""Default frame-sampling strategy for the Qwen video data path.

:func:`sample_frame_indices` decides which frames are fed to the model. The kit
default is uniform sampling across the whole clip. It is the swappable seam: the
encoders accept a ``sampler`` argument, so a recipe can inject a different
strategy (dynamic per-frame resolution, keyframe + cross-frame patch selection,
...) without editing the kit.
"""

from __future__ import annotations

import numpy as np

from .decoder import VideoMeta


def sample_frame_indices(meta: VideoMeta, num_frames: int) -> list[int]:
    """Uniformly sample ``num_frames`` frame indices across the whole video."""
    total = max(int(meta.total_num_frames), 1)
    count = max(1, min(int(num_frames), total))
    indices = np.linspace(0, total - 1, count).round().astype(int)
    return [int(index) for index in indices]
