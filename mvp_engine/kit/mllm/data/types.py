"""Shared MLLM data type definitions."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import torch


class ModelInputs(TypedDict):
    """Normalized multimodal tensors consumed by MLLM model forwards."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    labels: torch.Tensor
    pack_segment_ids: torch.Tensor
    source_sample_num: torch.Tensor
    num_input_tokens: torch.Tensor
    num_loss_tokens: torch.Tensor
    num_source_samples: torch.Tensor
    pixel_values: NotRequired[torch.Tensor]
    pixel_values_videos: NotRequired[torch.Tensor]
    image_grid_thw: NotRequired[torch.Tensor]
    video_grid_thw: NotRequired[torch.Tensor]
    position_ids: NotRequired[torch.Tensor]
    cu_seq_lens_q: NotRequired[torch.Tensor]
    cu_seq_lens_k: NotRequired[torch.Tensor]
    max_length_q: NotRequired[int]
    max_length_k: NotRequired[int]
    total_tokens: NotRequired[int]
    effective_tokens: NotRequired[int]
