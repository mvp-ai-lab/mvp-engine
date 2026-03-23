"""Shared type definitions for the minimal VLM recipe."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import torch


class TrainBatch(TypedDict):
    """Normalized multimodal batch consumed by the Qwen3-VL model."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pack_segment_ids: NotRequired[torch.Tensor]
    position_ids: NotRequired[torch.Tensor]
    pixel_values: NotRequired[torch.Tensor]
    image_grid_thw: NotRequired[torch.Tensor]
