"""Shared type definitions for the Qwen3 LM recipe."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import torch


class ModelInputs(TypedDict):
    """Normalized text tensors consumed by the Qwen3 causal LM."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pack_segment_ids: NotRequired[torch.Tensor]
    source_sample_num: NotRequired[torch.Tensor]
