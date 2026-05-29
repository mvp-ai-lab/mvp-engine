"""Shared type definitions for the Video VLM recipe."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import torch


class ModelInputs(TypedDict):
    """Normalized multimodal tensors consumed by the Qwen3-VL model."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pack_segment_ids: NotRequired[torch.Tensor]
    pixel_values: NotRequired[torch.Tensor]
    pixel_values_videos: NotRequired[torch.Tensor]
    image_grid_thw: NotRequired[torch.Tensor]
    video_grid_thw: NotRequired[torch.Tensor]
    patch_positions: NotRequired[torch.Tensor]
    source_sample_num: NotRequired[int | torch.Tensor]
