"""Unified visual-token video encoding helpers."""

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
class VideoEncodingResult:
    """Visual token sequence plus layout metadata for one video."""

    patch_values: torch.Tensor
    token_positions: torch.Tensor
    frame_grid_thw: torch.Tensor
    merge_sizes: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the visual-token sequence layout."""
        if self.patch_values.dim() != 4:
            raise ValueError("patch_values must have shape [N, C, patch_h, patch_w].")
        if int(self.patch_values.shape[0]) < 1:
            raise ValueError("patch_values must contain at least one visual token.")
        if self.token_positions.dim() != 2 or int(self.token_positions.shape[-1]) != 3:
            raise ValueError("token_positions must have shape [N, 3].")
        if int(self.patch_values.shape[0]) != int(self.token_positions.shape[0]):
            raise ValueError("patch_values and token_positions must have the same token count.")
        if self.frame_grid_thw.dim() != 2 or int(self.frame_grid_thw.shape[-1]) != 3:
            raise ValueError("frame_grid_thw must have shape [num_frames_or_segments, 3].")
        if self.merge_sizes.dim() != 1 or int(self.merge_sizes.shape[0]) != int(self.frame_grid_thw.shape[0]):
            raise ValueError("merge_sizes must have one value per frame_grid_thw row.")
        if torch.any(self.frame_grid_thw <= 0):
            raise ValueError("frame_grid_thw values must be positive.")
        if torch.any(self.merge_sizes <= 0):
            raise ValueError("merge_sizes values must be positive.")

    @property
    def visual_token_count(self) -> int:
        """Return the number of visual tokens produced by this encoding."""
        return int(self.patch_values.shape[0])

    @property
    def model_video_grid_thw(self) -> torch.Tensor:
        """Return one Qwen3-compatible video chunk row for placeholder insertion."""
        return torch.tensor([[1, self.visual_token_count, 1]], dtype=torch.long)

    def to_model_inputs(self) -> dict[str, torch.Tensor]:
        """Return tensors consumed by the collator/model adapter."""
        return {
            "pixel_values_videos": self.patch_values,
            "video_grid_thw": self.model_video_grid_thw,
            "video_token_positions": self.token_positions,
            "video_frame_grid_thw": self.frame_grid_thw,
            "video_merge_sizes": self.merge_sizes,
            "visual_token_count": torch.tensor(self.visual_token_count, dtype=torch.long),
        }


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


@dataclass(frozen=True)
class KeyframeLowresVideoConfig:
    """Geometry for dense variable-resolution keyframe/low-res video frames."""

    num_frames: int = 16
    full_frame_size: int = 224
    lowres_frame_size: int = 112
    patch_size: int = 14
    keyframe_interval: int = 4

    @property
    def full_grid_size(self) -> int:
        """Return the high-resolution spatial patch grid side length."""
        if self.full_frame_size % self.patch_size != 0:
            raise ValueError("keyframe full frame size must be divisible by patch size.")
        return self.full_frame_size // self.patch_size

    @property
    def lowres_grid_size(self) -> int:
        """Return the low-resolution spatial patch grid side length."""
        if self.lowres_frame_size % self.patch_size != 0:
            raise ValueError("keyframe low-res frame size must be divisible by patch size.")
        return self.lowres_frame_size // self.patch_size

    def is_keyframe(self, frame_index: int) -> bool:
        """Return whether a sampled frame should be encoded at full resolution."""
        return int(frame_index) % int(self.keyframe_interval) == 0

    def validate(self) -> None:
        """Validate variable-resolution frame geometry."""
        if self.num_frames < 1:
            raise ValueError("keyframe_lowres num_frames must be >= 1.")
        if self.keyframe_interval < 1:
            raise ValueError("keyframe_interval must be >= 1.")
        if self.lowres_frame_size > self.full_frame_size:
            raise ValueError("keyframe_lowres_frame_size must be <= video_frame_size.")
        _ = self.full_grid_size
        _ = self.lowres_grid_size

    def __fingerprint__(self) -> str:
        """Return a stable cache fingerprint for mvp_dataset map closures."""
        return (
            f"num_frames={self.num_frames}|full_frame_size={self.full_frame_size}|"
            f"lowres_frame_size={self.lowres_frame_size}|patch_size={self.patch_size}|"
            f"keyframe_interval={self.keyframe_interval}"
        )


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


def load_keyframe_lowres_video_frames(
    video: str | Path,
    config: KeyframeLowresVideoConfig,
) -> tuple[list[torch.Tensor], list[bool]]:
    """Decode sampled frames and resize keyframes/full frames independently."""
    config.validate()
    video_path = str(Path(video).expanduser().resolve())
    meta = probe_video(video_path)
    indices = sample_frame_indices(meta, config.num_frames)
    frames = decode_frames(video_path, indices)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

    resized_frames: list[torch.Tensor] = []
    keyframe_mask: list[bool] = []
    for frame_index, frame in enumerate(tensor):
        is_keyframe = config.is_keyframe(frame_index)
        frame_size = config.full_frame_size if is_keyframe else config.lowres_frame_size
        if tuple(frame.shape[-2:]) != (frame_size, frame_size):
            frame = tvF.resize(
                frame,
                [frame_size, frame_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        resized_frames.append(frame.contiguous())
        keyframe_mask.append(is_keyframe)
    return resized_frames, keyframe_mask


def process_video_with_dense_frames(
    video: str | Path,
    *,
    processor: Any,
    config: DenseVideoConfig,
) -> VideoEncodingResult:
    """Decode one uniformly sampled video into a dense visual-token sequence."""
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
    pixel_values = processed["pixel_values"].contiguous()
    patch_values = frames_to_patch_values(pixel_values, patch_size=config.patch_size)
    token_positions = dense_token_positions(
        num_frames=int(pixel_values.shape[0]),
        grid_h=config.grid_size,
        grid_w=config.grid_size,
    )

    return VideoEncodingResult(
        patch_values=patch_values,
        token_positions=token_positions,
        frame_grid_thw=torch.tensor([[1, config.grid_size, config.grid_size]] * config.num_frames, dtype=torch.long),
        merge_sizes=torch.ones(config.num_frames, dtype=torch.long),
    )


def process_video_with_keyframe_lowres(
    video: str | Path,
    *,
    processor: Any,
    config: KeyframeLowresVideoConfig,
) -> VideoEncodingResult:
    """Decode one video into full-res keyframes and complete low-res intermediate frames."""
    frames, keyframe_mask = load_keyframe_lowres_video_frames(video, config)
    image_processor = getattr(processor, "onevision_image_processor", getattr(processor, "image_processor", None))
    if image_processor is None:
        raise ValueError("processor must expose `onevision_image_processor` for video MLLM preprocessing.")
    if len(frames) != len(keyframe_mask):
        raise ValueError("keyframe_lowres frame count and keyframe mask length must match.")

    patch_values: list[torch.Tensor] = []
    token_positions: list[torch.Tensor] = []
    frame_grid_thw: list[list[int]] = []
    for frame_index, frame in enumerate(frames):
        pil_frame = tvF.to_pil_image(frame)
        processed = image_processor(
            images=[pil_frame],
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = processed["pixel_values"].contiguous()
        frame_patch_values = frames_to_patch_values(pixel_values, patch_size=config.patch_size)
        grid_h = int(pixel_values.shape[-2]) // int(config.patch_size)
        grid_w = int(pixel_values.shape[-1]) // int(config.patch_size)
        patch_values.append(frame_patch_values)
        token_positions.append(
            dense_frame_token_positions(
                frame_index=frame_index,
                grid_h=grid_h,
                grid_w=grid_w,
                coordinate_grid_h=config.full_grid_size,
                coordinate_grid_w=config.full_grid_size,
            )
        )
        frame_grid_thw.append([1, grid_h, grid_w])

    return VideoEncodingResult(
        patch_values=torch.cat(patch_values, dim=0),
        token_positions=torch.cat(token_positions, dim=0),
        frame_grid_thw=torch.tensor(frame_grid_thw, dtype=torch.long),
        merge_sizes=torch.ones(len(frame_grid_thw), dtype=torch.long),
    )


def frames_to_patch_values(frames: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    """Split frame tensors ``[T,C,H,W]`` into patch tensors ``[N,C,p,p]``."""
    if frames.dim() != 4:
        raise ValueError("frames must have shape [T, C, H, W].")
    num_frames, channels, height, width = frames.shape
    if channels != 3:
        raise ValueError("frames must contain RGB channels.")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("frame height/width must be divisible by patch_size.")

    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = frames.reshape(num_frames, channels, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
        num_frames * grid_h * grid_w, channels, patch_size, patch_size
    )
    return patches.contiguous()


def dense_token_positions(*, num_frames: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Return dense ``[t,h,w]`` positions for a full per-frame grid."""
    t = torch.arange(num_frames, dtype=torch.long).repeat_interleave(grid_h * grid_w)
    h = torch.arange(grid_h, dtype=torch.long).repeat_interleave(grid_w).repeat(num_frames)
    w = torch.arange(grid_w, dtype=torch.long).repeat(num_frames * grid_h)
    return torch.stack([t, h, w], dim=-1)


def dense_frame_token_positions(
    *,
    frame_index: int,
    grid_h: int,
    grid_w: int,
    coordinate_grid_h: int | None = None,
    coordinate_grid_w: int | None = None,
) -> torch.Tensor:
    """Return dense frame positions, optionally scaled onto a larger coordinate grid."""
    h = _dense_axis_positions(size=grid_h, coordinate_size=coordinate_grid_h)
    w = _dense_axis_positions(size=grid_w, coordinate_size=coordinate_grid_w)

    h_positions = h.repeat_interleave(grid_w)
    w_positions = w.repeat(grid_h)
    t_positions = torch.full((grid_h * grid_w,), int(frame_index), dtype=h_positions.dtype)
    return torch.stack([t_positions, h_positions, w_positions], dim=-1)


def _dense_axis_positions(*, size: int, coordinate_size: int | None = None) -> torch.Tensor:
    """Return dense axis positions, centered when one low-res token covers the axis."""
    if coordinate_size is None:
        return torch.arange(size, dtype=torch.long)
    if int(size) == 1:
        return torch.tensor([(int(coordinate_size) - 1) / 2.0], dtype=torch.float32)
    return torch.linspace(0, int(coordinate_size) - 1, int(size), dtype=torch.float32)
