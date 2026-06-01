"""OneVision-ready dense video encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.v2.functional as tvF
from torchvision.transforms import InterpolationMode

from .decoder import decode_frames, probe_video
from .sampling import sample_frame_indices


@dataclass(frozen=True)
class DenseVideoConfig:
    """Geometry for dense OneVision video frames."""

    num_frames: int = 16
    frame_size: int = 224
    patch_size: int = 14

    @property
    def grid_size(self) -> int:
        """Return the spatial patch grid side length."""
        if self.frame_size % self.patch_size != 0:
            raise ValueError("dense video frame size must be divisible by patch size.")
        return self.frame_size // self.patch_size

    def validate(self) -> None:
        """Validate dense frame geometry."""
        if self.num_frames < 1:
            raise ValueError("dense video num_frames must be >= 1.")
        _ = self.grid_size

    def __fingerprint__(self) -> str:
        """Return a stable cache fingerprint for mvp_dataset map closures."""
        return f"num_frames={self.num_frames}|frame_size={self.frame_size}|patch_size={self.patch_size}"


def load_dense_video_frames(video: str | Path, config: DenseVideoConfig) -> torch.Tensor:
    """Decode uniformly sampled frames into a fixed-size ``[T,C,H,W]`` tensor."""
    config.validate()
    video_path = str(Path(video).expanduser().resolve())
    meta = probe_video(video_path)
    indices = sample_frame_indices(meta, config.num_frames)
    frames = decode_frames(video_path, indices)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    if tuple(tensor.shape[-2:]) != (config.frame_size, config.frame_size):
        tensor = tvF.resize(
            tensor,
            [config.frame_size, config.frame_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    return tensor


def process_video_with_dense_frames(
    video: str | Path,
    *,
    processor: Any,
    config: DenseVideoConfig,
) -> dict[str, torch.Tensor]:
    """Decode one uniformly sampled video into OneVision-ready dense tensors."""
    frames = load_dense_video_frames(video, config)
    image_processor = getattr(processor, "onevision_image_processor", getattr(processor, "image_processor", None))
    if image_processor is None:
        raise ValueError("processor must expose `onevision_image_processor` for video MLLM preprocessing.")

    pil_frames = [tvF.to_pil_image(frame) for frame in frames]
    processed = image_processor(
        images=pil_frames,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    pixel_values = processed["pixel_values"].unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

    return {
        "pixel_values_videos": pixel_values,
        "video_grid_thw": torch.tensor([[config.num_frames, config.grid_size, config.grid_size]], dtype=torch.long),
    }
