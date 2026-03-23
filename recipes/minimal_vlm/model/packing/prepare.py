"""Batch preparation helpers for packed minimal_vlm training."""

from __future__ import annotations

from typing import Any

import torch

from .masks import build_packed_block_causal_mask
from .qwen3_vl import build_qwen3_vl_packed_position_ids


def prepare_packed_model_inputs(
    batch: dict[str, Any],
    *,
    model_config: Any,
    attn_implementation: str | None,
    mask_dtype: torch.dtype,
) -> dict[str, Any]:
    """Convert packed batch metadata into model-ready attention inputs."""
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
        batch["attention_mask"] = pack_segment_ids
    else:
        batch["attention_mask"] = build_packed_block_causal_mask(pack_segment_ids, dtype=mask_dtype)
    return batch


__all__ = ["prepare_packed_model_inputs"]
