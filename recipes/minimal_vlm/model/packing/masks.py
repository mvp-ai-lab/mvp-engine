"""Packed attention-mask builders."""

from __future__ import annotations

import torch


def build_packed_block_causal_mask(
    pack_segment_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 4D additive mask that isolates packed samples."""
    if pack_segment_ids.ndim != 2:
        raise ValueError(f"Expected 2D pack_segment_ids, got shape {tuple(pack_segment_ids.shape)}.")

    batch_size, sequence_length = pack_segment_ids.shape
    token_positions = torch.arange(sequence_length, device=pack_segment_ids.device)
    causal_mask = token_positions.unsqueeze(0) <= token_positions.unsqueeze(1)

    valid_tokens = pack_segment_ids.ne(0)
    same_segment = pack_segment_ids.unsqueeze(-1) == pack_segment_ids.unsqueeze(-2)
    allowed = valid_tokens.unsqueeze(-1) & valid_tokens.unsqueeze(-2) & same_segment & causal_mask.unsqueeze(0)

    min_dtype = torch.finfo(dtype).min
    attention_mask = torch.full(
        (batch_size, 1, sequence_length, sequence_length),
        min_dtype,
        dtype=dtype,
        device=pack_segment_ids.device,
    )
    attention_mask.masked_fill_(allowed.unsqueeze(1), 0)
    return attention_mask


__all__ = ["build_packed_block_causal_mask"]
