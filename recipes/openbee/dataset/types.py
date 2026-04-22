"""Shared type definitions for the OpenBee recipe."""

from __future__ import annotations

try:
    from typing import NotRequired, TypedDict
except ImportError:  # Python < 3.11
    from typing import TypedDict
    from typing_extensions import NotRequired

import torch


class ModelInputs(TypedDict):
    """Normalized multimodal tensors consumed by the Qwen3-VL model."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pack_segment_ids: NotRequired[torch.Tensor]
    pixel_values: NotRequired[torch.Tensor]
    image_grid_thw: NotRequired[torch.Tensor]
