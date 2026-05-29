"""Batch preparation helpers for packed Qwen3 LM training."""

from __future__ import annotations

from typing import Any

import torch


def prepare_packed_model_inputs(
    batch: dict[str, Any],
    *,
    attn_implementation: str | None,
    mask_dtype: torch.dtype,
) -> dict[str, Any]:
    """Convert packed batch metadata into model-ready attention inputs."""
    pack_segment_ids = batch.pop("pack_segment_ids", None)
    if pack_segment_ids is None:
        return batch

    batch["position_ids"] = build_qwen3_packed_position_ids(
        input_ids=batch["input_ids"],
        pack_segment_ids=pack_segment_ids,
    )

    if attn_implementation == "flash_attention_2":
        batch["attention_mask"] = pack_segment_ids
    else:
        batch["attention_mask"] = _build_packed_block_causal_mask(pack_segment_ids, dtype=mask_dtype)

    return batch


def build_qwen3_packed_position_ids(
    *,
    input_ids: torch.Tensor,
    pack_segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Build cumulative text RoPE position ids for packed Qwen3 samples."""
    if input_ids.ndim != 2:
        raise ValueError(f"Expected 2D input_ids, got shape {tuple(input_ids.shape)}.")
    if pack_segment_ids.shape != input_ids.shape:
        raise ValueError(
            "pack_segment_ids must have the same shape as input_ids, "
            f"got {tuple(pack_segment_ids.shape)} vs {tuple(input_ids.shape)}."
        )

    batch_size, sequence_length = input_ids.shape
    token_positions = torch.arange(sequence_length, device=input_ids.device, dtype=torch.long)
    position_ids = token_positions.view(1, -1).expand(batch_size, -1).clone()
    position_ids.masked_fill_(pack_segment_ids.eq(0), 1)
    return position_ids


def _build_packed_block_causal_mask(
    pack_segment_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 4D additive mask that isolates packed samples for eager/SDPA backends."""
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
