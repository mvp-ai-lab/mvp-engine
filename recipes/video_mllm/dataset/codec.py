"""On-the-fly codec patchification helpers for the video MLLM recipe.

Ported from the Video VLM recipe. The only change is the frame source: this
module uses the recipe's PyAV decoder (``decoder.probe_video`` /
``decoder.decode_frames``) instead of ``decord``, which has no Python 3.12
wheel. All tensor/patch logic and the cv_reader residual path are unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.v2.functional as tvF
from torchvision.transforms import InterpolationMode

from .decoder import decode_frames, probe_video
from .video_encoding import VideoEncodingResult, frames_to_patch_values


@dataclass(frozen=True)
class CodecPatchConfig:
    """Geometry and decoder settings for OneVision-style codec patches."""

    num_frames: int = 64
    packed_frames: int = 8
    frame_size: int = 224
    patch_size: int = 14
    k_keep: int = 2048
    hevc_decoder_bin: str | None = None
    cv_reader_required: bool = True

    @property
    def grid_size(self) -> int:
        """Return the spatial patch grid side length."""
        if self.frame_size % self.patch_size != 0:
            raise ValueError("codec frame size must be divisible by patch size.")
        return self.frame_size // self.patch_size

    @property
    def patches_per_frame(self) -> int:
        """Return the number of patches in one packed frame."""
        return self.grid_size * self.grid_size

    def validate(self) -> None:
        """Validate the packed-frame token budget."""
        expected = self.packed_frames * self.patches_per_frame
        if self.k_keep != expected:
            raise ValueError(f"codec_k_keep must be {expected} for the configured packed geometry.")

    def __fingerprint__(self) -> str:
        """Return a stable cache fingerprint for mvp_dataset map closures."""
        return (
            f"num_frames={self.num_frames}|packed_frames={self.packed_frames}|"
            f"frame_size={self.frame_size}|patch_size={self.patch_size}|"
            f"k_keep={self.k_keep}|hevc_decoder_bin={self.hevc_decoder_bin or ''}|"
            f"cv_reader_required={int(self.cv_reader_required)}"
        )


def mask_by_residual_topk(residuals: torch.Tensor, k_keep: int, patch_size: int) -> torch.Tensor:
    """Select flattened ``t,h,w`` patch indices with largest residual magnitude.

    Args:
        residuals: Tensor shaped ``[B, 1, T, H, W]``.
        k_keep: Number of visible patches to keep.
        patch_size: Spatial patch size.

    Returns:
        Sorted visible indices shaped ``[B, k_keep]``.
    """
    if residuals.dim() != 5 or residuals.size(1) != 1:
        raise ValueError("residuals must have shape [B, 1, T, H, W].")

    batch, _, frames, height, width = residuals.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("residual height/width must be divisible by patch_size.")

    grid_h = height // patch_size
    grid_w = width // patch_size
    total_patches = frames * grid_h * grid_w
    keep = max(0, min(int(k_keep), int(total_patches)))

    scores = residuals.abs().squeeze(1)
    scores = scores.reshape(batch, frames, grid_h, patch_size, grid_w, patch_size)
    scores = scores.sum(dim=(3, 5)).reshape(batch, total_patches)
    topk = torch.topk(scores, k=keep, dim=1, largest=True, sorted=False).indices
    return torch.sort(topk, dim=1).values


def indices_to_patch_positions(indices: torch.Tensor, *, grid_h: int, grid_w: int) -> torch.Tensor:
    """Convert flattened codec patch indices into explicit ``[t, h, w]`` positions."""
    if indices.dim() != 1:
        raise ValueError("indices must be a 1D tensor.")
    patches_per_frame = int(grid_h) * int(grid_w)
    t = indices // patches_per_frame
    rem = indices % patches_per_frame
    h = rem // int(grid_w)
    w = rem % int(grid_w)
    return torch.stack([t, h, w], dim=-1).to(dtype=torch.long)


def pack_video_patches(video: torch.Tensor, visible_indices: torch.Tensor, config: CodecPatchConfig) -> torch.Tensor:
    """Pack selected sparse patches into dense codec frames.

    Args:
        video: RGB frame tensor shaped ``[T, C, H, W]``.
        visible_indices: Flattened selected patch indices shaped ``[k_keep]``.
        config: Codec geometry.

    Returns:
        Packed RGB frames shaped ``[packed_frames, C, H, W]``.
    """
    config.validate()
    if video.dim() != 4:
        raise ValueError("video must have shape [T, C, H, W].")
    frames, channels, height, width = video.shape
    if channels != 3:
        raise ValueError("video must contain RGB frames.")
    if height != config.frame_size or width != config.frame_size:
        raise ValueError("video frames must already match codec_frame_size.")
    if int(visible_indices.numel()) != config.k_keep:
        raise ValueError("visible_indices length must equal codec_k_keep.")

    grid = config.grid_size
    patches = video.reshape(frames, channels, grid, config.patch_size, grid, config.patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
        frames * grid * grid, channels, config.patch_size, config.patch_size
    )
    selected = patches.index_select(0, visible_indices.to(device=video.device, dtype=torch.long))
    packed = selected.reshape(config.packed_frames, grid, grid, channels, config.patch_size, config.patch_size)
    packed = packed.permute(0, 3, 1, 4, 2, 5).reshape(
        config.packed_frames, channels, config.frame_size, config.frame_size
    )
    return packed


def _sample_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    """Return uniformly sampled frame ids, padding short videos with their last frame."""
    if frame_count <= 0:
        raise ValueError("video has no frames.")
    if frame_count >= int(num_frames):
        return torch.linspace(0, frame_count - 1, int(num_frames)).long().tolist()
    return list(range(frame_count)) + [frame_count - 1] * (int(num_frames) - frame_count)


def _load_video_frames(video_path: str | os.PathLike[str], config: CodecPatchConfig) -> torch.Tensor:
    """Load uniformly sampled RGB frames with the recipe's PyAV decoder.

    Returns a ``[T, C, H, W]`` uint8 RGB tensor of ``config.num_frames`` frames,
    resized to ``config.frame_size`` so it can feed ``pack_video_patches``.
    """
    frame_count = _probe_video_frame_count(video_path)
    frame_indices = _sample_frame_indices(frame_count, config.num_frames)

    # decode_frames returns an (T, H, W, 3) uint8 RGB array and repeats the last
    # decoded frame for any out-of-range index, so the count always matches.
    frames = decode_frames(str(video_path), frame_indices)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    if tuple(tensor.shape[-2:]) != (config.frame_size, config.frame_size):
        tensor = tvF.resize(
            tensor,
            [config.frame_size, config.frame_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    return tensor


def _frame_difference_residuals_from_frames(frames: torch.Tensor) -> torch.Tensor:
    """Build fallback residuals as grayscale inter-frame differences of decoded frames.

    Args:
        frames: Decoded RGB frame tensor shaped ``[T, C, H, W]``.

    Returns:
        Residual tensor shaped ``[1, 1, T, H, W]``.
    """
    gray = frames.to(dtype=torch.float32).mean(dim=1)
    residuals = torch.zeros_like(gray)
    residuals[1:] = gray[1:] - gray[:-1]
    return residuals.unsqueeze(0).unsqueeze(0)


def _load_frame_difference_residuals(
    video_path: str | os.PathLike[str],
    config: CodecPatchConfig,
    frames: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build fallback residuals from grayscale frame differences.

    When ``frames`` is provided (the already-decoded clip), it is reused to avoid a
    second decode; otherwise the clip is decoded here.
    """
    if frames is None:
        frames = _load_video_frames(video_path, config)
    return _frame_difference_residuals_from_frames(frames)


def _probe_video_codec(video_path: str | os.PathLike[str]) -> str | None:
    """Return the first video stream codec name using ffprobe when available."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") or []
        if not streams:
            return None
        codec_name = streams[0].get("codec_name")
        return str(codec_name).lower() if codec_name else None
    except Exception:
        return None


def _probe_video_frame_count(video_path: str | os.PathLike[str]) -> int:
    """Return the video frame count using the recipe's PyAV probe."""
    try:
        return int(probe_video(str(video_path)).total_num_frames)
    except Exception as exc:
        raise ValueError(f"could not determine video frame count: {video_path}") from exc


def _resolve_cv_reader_api():
    """Import OneVision's patched-FFmpeg cv_reader API."""
    try:
        return import_module("cv_reader.api")
    except Exception as exc:
        raise RuntimeError(
            "video MLLM codec residuals require `cv_reader` for H.264/H.265 support. "
            "Install it from OneVision-Encoder/llava_next/Compressed_Video_Reader with the mvp-engine venv active "
            "(for example: `cd .../Compressed_Video_Reader && bash install.sh`)."
        ) from exc


def _residual_y_from_cv_reader_frame(frame: dict[str, Any]) -> Any:
    """Extract a Y-plane residual map from one cv_reader frame dict."""
    if "residual_y" in frame:
        return frame["residual_y"]

    residual = frame.get("residual")
    if residual is None:
        raise ValueError("cv_reader frame does not contain `residual_y` or `residual`.")

    tensor = torch.as_tensor(residual)
    if tensor.dim() == 2:
        return residual
    if tensor.dim() == 3 and tensor.shape[-1] >= 3:
        # Approximate luma from an HWC residual image. Channel order is not
        # critical for saliency because we only use residual magnitude.
        return tensor[..., :3].to(dtype=torch.float32).mean(dim=-1).numpy()
    if tensor.dim() == 3 and tensor.shape[0] in {1, 3}:
        return tensor.to(dtype=torch.float32).mean(dim=0).numpy()
    raise ValueError(f"unexpected cv_reader residual shape: {tuple(tensor.shape)}")


def _fill_missing_residuals(residuals: list[Any | None]) -> list[Any]:
    """Fill missing cv_reader callback results by carrying nearest decoded residuals."""
    last = None
    for index, residual in enumerate(residuals):
        if residual is None:
            if last is not None:
                residuals[index] = last
        else:
            last = residual

    next_seen = None
    for index in range(len(residuals) - 1, -1, -1):
        residual = residuals[index]
        if residual is None:
            if next_seen is not None:
                residuals[index] = next_seen
        else:
            next_seen = residual

    return [residual for residual in residuals if residual is not None]


def _load_cv_reader_residual_arrays(
    video_path: str | os.PathLike[str],
    *,
    frame_indices: list[int],
    cv_api: Any | None = None,
) -> list[Any]:
    """Load selected residual maps from cv_reader for H.264/H.265 videos."""
    if not frame_indices:
        return []
    cv_api = cv_api or _resolve_cv_reader_api()

    read_video_cb = getattr(cv_api, "read_video_cb", None)
    if callable(read_video_cb):
        positions_by_frame: dict[int, list[int]] = {}
        for output_index, frame_id in enumerate(frame_indices):
            positions_by_frame.setdefault(int(frame_id), []).append(output_index)

        outputs: list[Any | None] = [None] * len(frame_indices)

        def _callback(frame: dict[str, Any]) -> bool:
            frame_id = int(frame.get("frame_idx", -1))
            positions = positions_by_frame.get(frame_id)
            if positions:
                output_index = positions.pop(0)
                outputs[output_index] = _residual_y_from_cv_reader_frame(frame)
            return any(positions for positions in positions_by_frame.values())

        max_frames = int(max(frame_indices)) + 1
        try:
            read_video_cb(str(video_path), _callback, 0, max_frames, [int(index) for index in frame_indices])
        except TypeError:
            read_video_cb(str(video_path), _callback, 0, max_frames)
        return _fill_missing_residuals(outputs)

    read_video = getattr(cv_api, "read_video", None)
    if not callable(read_video):
        raise RuntimeError("cv_reader.api does not expose read_video_cb() or read_video().")

    frames = read_video(str(video_path), 0, int(max(frame_indices)) + 1)
    if not isinstance(frames, (list, tuple)):
        frames = list(frames)
    residuals: list[Any] = []
    for frame_id in frame_indices:
        if not frames:
            break
        clamped = max(0, min(int(frame_id), len(frames) - 1))
        residuals.append(_residual_y_from_cv_reader_frame(frames[clamped]))
    return residuals


def _residual_arrays_to_tensor(residual_arrays: list[Any], config: CodecPatchConfig) -> torch.Tensor:
    """Normalize cv_reader residual arrays into ``[1,1,T,H,W]`` tensors."""
    residuals: list[torch.Tensor] = []
    for y in residual_arrays:
        if hasattr(y, "flags") and not y.flags.writeable:
            y = y.copy()
        tensor = torch.as_tensor(y).squeeze()
        if tensor.dim() != 2:
            raise ValueError(f"unexpected residual shape: {tuple(tensor.shape)}")
        tensor = tensor.to(dtype=torch.float32).unsqueeze(0)
        if tuple(tensor.shape[-2:]) != (config.frame_size, config.frame_size):
            tensor = tvF.resize(
                tensor,
                [config.frame_size, config.frame_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        residuals.append(tensor.squeeze(0) - 128.0)
        if len(residuals) >= config.num_frames:
            break

    if not residuals:
        raise ValueError("cv_reader decoded zero residual frames.")
    while len(residuals) < config.num_frames:
        residuals.append(torch.zeros_like(residuals[-1]))
    stacked = torch.stack(residuals[: config.num_frames], dim=0)
    return stacked.unsqueeze(0).unsqueeze(0)


def _load_residuals(
    video_path: str | os.PathLike[str],
    config: CodecPatchConfig,
    frames: torch.Tensor | None = None,
) -> torch.Tensor:
    """Load codec residuals as ``[1, 1, T, H, W]`` for saliency scoring.

    ``frames`` is the already-decoded clip (``[T, C, H, W]``); when provided it is
    reused by the frame-difference fallback to avoid a second decode. The cv_reader
    (true-residual) path is unaffected and always reads its own residual stream.
    """
    codec_name = _probe_video_codec(video_path)
    if codec_name is not None and codec_name not in {"h264", "hevc", "h265"}:
        if config.cv_reader_required:
            raise ValueError(
                "cv_reader codec residuals require an H.264 or H.265/HEVC video, "
                f"but {video_path} uses codec '{codec_name}'."
            )
        return _load_frame_difference_residuals(video_path, config, frames=frames)

    try:
        frame_indices = _sample_frame_indices(_probe_video_frame_count(video_path), config.num_frames)
        residual_arrays = _load_cv_reader_residual_arrays(video_path, frame_indices=frame_indices)
        return _residual_arrays_to_tensor(residual_arrays, config)
    except Exception as exc:
        if config.cv_reader_required:
            raise RuntimeError(
                "Failed to decode H.264/H.265 codec residuals with cv_reader. "
                f"video={video_path}, codec={codec_name or '<unknown>'}. "
                "Install/verify OneVision-Encoder/llava_next/Compressed_Video_Reader in the active venv."
            ) from exc
        return _load_frame_difference_residuals(video_path, config, frames=frames)


def process_video_with_codec(
    video: str | os.PathLike[str],
    *,
    processor: Any,
    config: CodecPatchConfig,
) -> VideoEncodingResult:
    """Decode one video and return a sparse codec visual-token sequence."""
    config.validate()
    video_path = str(Path(video).expanduser().resolve())
    frames = _load_video_frames(video_path, config)
    # Reuse the already-decoded frames for the frame-difference fallback residuals
    # (no second decode); the cv_reader true-residual path reads its own stream.
    residuals = _load_residuals(video_path, config, frames=frames)
    visible_indices = mask_by_residual_topk(residuals, config.k_keep, config.patch_size)[0].cpu()
    packed_frames = pack_video_patches(frames, visible_indices, config)
    positions = indices_to_patch_positions(visible_indices, grid_h=config.grid_size, grid_w=config.grid_size)

    image_processor = getattr(processor, "onevision_image_processor", getattr(processor, "image_processor", None))
    if image_processor is None:
        raise ValueError("processor must expose `onevision_image_processor` for video MLLM codec preprocessing.")
    pil_frames = [tvF.to_pil_image(frame) for frame in packed_frames]
    processed = image_processor(
        images=pil_frames,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    packed_pixel_values = processed["pixel_values"].contiguous()
    patch_values = frames_to_patch_values(packed_pixel_values, patch_size=config.patch_size)

    return VideoEncodingResult(
        patch_values=patch_values,
        token_positions=positions,
        frame_grid_thw=torch.tensor([[1, config.grid_size, config.grid_size]] * config.num_frames, dtype=torch.long),
        merge_sizes=torch.ones(config.num_frames, dtype=torch.long),
    )
