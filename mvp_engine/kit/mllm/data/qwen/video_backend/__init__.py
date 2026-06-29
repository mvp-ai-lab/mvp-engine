"""Video decode/sampling/encoding backend for the Qwen video data path.

Model-agnostic mechanism: PyAV decode, frame sampling, and OneVision visual-token
encoding. The codec-patch strategy and its external decode dependency stay in the
recipe and are injected into the media handler.
"""

from .decoder import VideoMeta, decode_frames, probe_video
from .image_encoding import process_image_as_frame, read_image
from .sampling import sample_frame_indices
from .video_encoding import (
    DenseVideoConfig,
    FrameSampler,
    KeyframeLowresVideoConfig,
    VideoEncodingResult,
    dense_frame_token_positions,
    dense_token_positions,
    frames_to_patch_values,
    load_dense_video_frames,
    load_keyframe_lowres_video_frames,
    process_video_with_dense_frames,
    process_video_with_keyframe_lowres,
)

__all__ = [
    "DenseVideoConfig",
    "FrameSampler",
    "KeyframeLowresVideoConfig",
    "VideoEncodingResult",
    "VideoMeta",
    "decode_frames",
    "dense_frame_token_positions",
    "dense_token_positions",
    "frames_to_patch_values",
    "load_dense_video_frames",
    "load_keyframe_lowres_video_frames",
    "probe_video",
    "process_image_as_frame",
    "process_video_with_dense_frames",
    "process_video_with_keyframe_lowres",
    "read_image",
    "sample_frame_indices",
]
