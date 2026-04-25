"""Shared type definitions for the OpenBee recipe."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import torch

SOURCE_SAMPLE_COUNT_KEY = "source_sample_count"


class ModelInputs(TypedDict):
    """Normalized multimodal tensors consumed by the Qwen3-VL model."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pack_segment_ids: NotRequired[torch.Tensor]
    pixel_values: NotRequired[torch.Tensor]
    image_grid_thw: NotRequired[torch.Tensor]
    source_sample_count: NotRequired[int | torch.Tensor]
