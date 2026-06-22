"""Shared text-LM data type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

import torch


@dataclass(frozen=True, slots=True)
class LLMSegment:
    """One ordered text segment and whether it contributes supervised loss."""

    type: str
    loss: bool
    value: Any


class ModelInputs(TypedDict):
    """Token tensors consumed by a text-only LM forward."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    labels: torch.Tensor
    pack_segment_ids: torch.Tensor
    source_sample_num: torch.Tensor
    num_input_tokens: torch.Tensor
    num_loss_tokens: torch.Tensor
    num_source_samples: torch.Tensor
    position_ids: NotRequired[torch.Tensor]
    cu_seq_lens_q: NotRequired[torch.Tensor]
    cu_seq_lens_k: NotRequired[torch.Tensor]
    max_length_q: NotRequired[int]
    max_length_k: NotRequired[int]
    total_tokens: NotRequired[int]
    effective_tokens: NotRequired[int]
