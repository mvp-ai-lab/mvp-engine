"""Shared type definitions for the video MLLM recipe."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import torch


class ModelInputs(TypedDict):
    """Normalized multimodal tensors consumed by the Qwen3-VL model."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values_videos: NotRequired[torch.Tensor]
    video_grid_thw: NotRequired[torch.Tensor]
    video_token_positions: NotRequired[torch.Tensor]
    video_token_counts: NotRequired[torch.Tensor]
    video_frame_grid_thw: NotRequired[torch.Tensor]
    video_merge_sizes: NotRequired[torch.Tensor]
    video_frame_counts: NotRequired[torch.Tensor]
    visual_token_count: NotRequired[torch.Tensor]
    total_tokens: NotRequired[int]
    effective_tokens: NotRequired[int]
