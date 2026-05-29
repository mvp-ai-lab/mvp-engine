"""Batch preparation helpers for packed Video VLM training."""

from __future__ import annotations

from typing import Any

import torch

from .qwen3_vl import build_qwen3_vl_packed_position_ids


def prepare_packed_model_inputs(
    batch: dict[str, Any],
    *,
    model_config: Any,
    attn_implementation: str | None,
    mask_dtype: torch.dtype,
) -> dict[str, Any]:
    """Convert packed batch metadata into model-ready attention inputs.

    Pops ``pack_segment_ids`` from the batch and replaces it with:
    - ``position_ids``: cumulative RoPE positions across packed segments (always)
    - ``attention_mask``: segment-id tensor for FA2, or a 4D additive mask for other backends

    If ``pack_segment_ids`` is absent the batch is returned unchanged.
    """
    pack_segment_ids = batch.pop("pack_segment_ids", None)
    if pack_segment_ids is None:
        return batch

    batch["position_ids"] = build_qwen3_vl_packed_position_ids(
        input_ids=batch["input_ids"],
        pack_segment_ids=pack_segment_ids,
        image_grid_thw=batch.get("image_grid_thw"),
        model_config=model_config,
    )

    if attn_implementation == "flash_attention_2":
        # FA2 path: pass segment ids directly; apply_packed_fa2_patch() makes the HF
        # FA2 utils interpret integer segment-id masks as packed cu_seqlens boundaries.
        batch["attention_mask"] = pack_segment_ids
    else:
        # Eager / SDPA path: build a standard 4D additive causal mask that blocks
        # cross-segment attention.
        batch["attention_mask"] = _build_packed_block_causal_mask(pack_segment_ids, dtype=mask_dtype)

    return batch


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
