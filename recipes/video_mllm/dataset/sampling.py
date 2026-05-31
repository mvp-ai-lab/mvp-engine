"""Frame-sampling strategy for the video MLLM recipe.

This is the swappable seam: :func:`sample_frame_indices` decides which frames are
fed to the model. The default is uniform sampling across the whole clip. To try
another strategy (dynamic per-frame resolution, keyframe + cross-frame patch
selection, ...), edit this function and the matching ``preprocess.py`` /
model-side code, following the ``skills/data/video-frame-sampling`` skill. Keep
the strategy concrete here rather than adding a registry or config dispatch.
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
