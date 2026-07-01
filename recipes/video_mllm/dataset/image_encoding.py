"""Single-image encoding for OneVision image-text alignment.

Each image is encoded as one OneVision frame so it flows through the exact same
validated visual patch-sequence path as video (``get_video_features``) with no
model changes. Used by image-modality stages such as projector alignment
(e.g. OpenBee Stage 1 image-text data).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from torchvision.transforms import InterpolationMode

from .video_encoding import (
    DenseVideoConfig,
    VideoEncodingResult,
    dense_token_positions,
    frames_to_patch_values,
)


def read_image(image: Any, *, image_root: str | None = None) -> Image.Image:
    """Decode one resolved image reference into an RGB PIL image.

    Handles raw bytes, ``{"bytes" | "path"}`` records, filesystem paths
    (optionally resolved under ``image_root``), and already-decoded PIL images.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    if isinstance(image, dict):
        data = image.get("bytes")
        if isinstance(data, (bytes, bytearray, memoryview)):
            return read_image(bytes(data))
        path = image.get("path")
        if isinstance(path, str) and path:
            return read_image(path, image_root=image_root)
        raise ValueError("contains an invalid image record.")
    if isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as decoded:
            return decoded.convert("RGB")
    if not isinstance(image, str) or not image:
        raise ValueError(f"contains an invalid image value: {type(image).__name__}.")

    path = Path(image).expanduser()
    if image_root is not None and not path.is_absolute():
        path = Path(image_root).expanduser() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"references missing image: {path}")
    with Image.open(path) as opened:
        return opened.convert("RGB").copy()


def process_image_as_frame(
    image: Any,
    *,
    processor: Any,
    config: DenseVideoConfig,
    image_root: str | None = None,
) -> VideoEncodingResult:
    """Encode one still image into a single-frame OneVision visual-token sequence."""
    config.validate()
    image_processor = getattr(processor, "onevision_image_processor", getattr(processor, "image_processor", None))
    if image_processor is None:
        raise ValueError("processor must expose `onevision_image_processor` for image MLLM preprocessing.")

    frame = tvF.pil_to_tensor(read_image(image, image_root=image_root))
    if tuple(frame.shape[-2:]) != (config.frame_size, config.frame_size):
        frame = tvF.resize(
            frame,
            [config.frame_size, config.frame_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    processed = image_processor(
        images=[tvF.to_pil_image(frame)],
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    pixel_values = processed["pixel_values"].contiguous()
    patch_values = frames_to_patch_values(pixel_values, patch_size=config.patch_size)
    token_positions = dense_token_positions(num_frames=1, grid_h=config.grid_size, grid_w=config.grid_size)

    return VideoEncodingResult(
        patch_values=patch_values,
        token_positions=token_positions,
        frame_grid_thw=torch.tensor([[1, config.grid_size, config.grid_size]], dtype=torch.long),
        merge_sizes=torch.ones(1, dtype=torch.long),
    )
