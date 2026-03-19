"""Shared type definitions for the minimal VLM recipe."""

from __future__ import annotations

from typing import TypedDict

import torch


class TrainBatch(TypedDict):
    """Normalized multimodal batch consumed by the Qwen3-VL model."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
