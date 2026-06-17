"""Shared MLLM data type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

import torch


@dataclass(frozen=True, slots=True)
class MLLMMediaSlot:
    """A media reference position attached to one normalized MLLM sample.

    Attributes:
        media_id: Stable id referenced by media segments.
        media_type: Media type name, such as ``"image"``.
        field: Raw sample field that stores the media value.
        index: Optional index inside the raw field value when it is a sequence.
        metadata: Optional schema-provided metadata, such as original media size.
    """

    media_id: str
    media_type: str
    field: str
    index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MLLMSegment:
    """One ordered text or media segment and whether it contributes supervised loss.

    Attributes:
        type: Segment type. ``"text"`` stores literal text; other values name media types.
        loss: Whether tokens produced from this segment should be training labels.
        value: Text content for text segments, or media id for media segments before rendering.
    """

    type: str
    loss: bool
    value: Any


class ModelInputs(TypedDict):
    """Normalized multimodal tensors consumed by MLLM model forwards.

    Required keys are produced by the standard datakit pipeline. Optional keys are
    model- or modality-specific fields added by media handlers or downstream model
    input preparation.
    """

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
