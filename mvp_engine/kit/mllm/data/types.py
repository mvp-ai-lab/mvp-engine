"""Shared MLLM data type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, TypedDict

import torch


@dataclass(frozen=True, slots=True)
class CanonicalMedia:
    """Normalized media reference attached to one canonical MLLM sample."""

    type: Literal["image", "video", "audio"]
    value: Any
    size: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CanonicalMLLMSample:
    """Normalized MLLM sample before model-family tokenization."""

    messages: list[dict[str, Any]]
    media: list[CanonicalMedia]
    metadata: dict[str, Any] = field(default_factory=dict)


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
