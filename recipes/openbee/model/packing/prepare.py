"""Prepare canonical packed OpenBee batches for Qwen3-VL forward."""

from typing import Any

import torch

from mvp_engine.kit.mllm.data import build_packed_block_causal_mask

from .qwen3_vl import build_qwen3_vl_packed_position_ids


def prepare_packed_model_inputs(
    batch: dict[str, Any],
    *,
    model_config: Any,
    attn_implementation: str | None,
    mask_dtype: torch.dtype,
) -> dict[str, Any]:
    """Convert DataKit canonical packed metadata into Qwen3-VL model inputs."""
    pack_segment_ids = batch.get("pack_segment_ids")
    if pack_segment_ids is None:
        raise ValueError("Packed OpenBee batches must include pack_segment_ids.")

    batch.pop("source_sample_num", None)
    batch.pop("num_input_tokens", None)
    batch.pop("num_loss_tokens", None)
    batch.pop("num_source_samples", None)

    batch["position_ids"] = build_qwen3_vl_packed_position_ids(
        input_ids=batch["input_ids"],
        pack_segment_ids=pack_segment_ids,
        image_grid_thw=batch.get("image_grid_thw"),
        model_config=model_config,
    )

    if attn_implementation == "flash_attention_2":
        batch["attention_mask"] = None
        batch.update(build_packed_fa2_varlen_kwargs(pack_segment_ids))
    else:
        batch["attention_mask"] = build_packed_block_causal_mask(pack_segment_ids, dtype=mask_dtype)

    return batch


def build_packed_fa2_varlen_kwargs(pack_segment_ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
    """Build FlashAttention varlen kwargs from packed segment ids."""
    if pack_segment_ids.ndim != 2:
        raise ValueError(f"Expected 2D pack_segment_ids, got shape {tuple(pack_segment_ids.shape)}.")

    segment_lengths = []
    for row in pack_segment_ids:
        if row.numel() == 0:
            continue
        valid_length = int(row.ne(0).sum().item())
        if valid_length <= 0:
            raise ValueError("Each packed FlashAttention row must contain at least one non-padding token.")
        if bool(row[:valid_length].eq(0).any().item()) or bool(row[valid_length:].ne(0).any().item()):
            raise ValueError("Packed FlashAttention padding must be a single zero-valued suffix.")

        starts = torch.cat(
            [
                torch.zeros(1, device=row.device, dtype=torch.long),
                torch.nonzero(row[1:] != row[:-1], as_tuple=False).flatten() + 1,
            ]
        )
        ends = torch.cat(
            [
                starts[1:],
                torch.tensor([row.numel()], device=row.device, dtype=torch.long),
            ]
        )
        segment_lengths.append(ends - starts)

    if not segment_lengths:
        raise ValueError("pack_segment_ids must contain at least one token.")

    seqlens = torch.cat(segment_lengths).to(dtype=torch.int32)
    cu_seqlens = torch.zeros(seqlens.numel() + 1, device=pack_segment_ids.device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    if int(cu_seqlens[-1].item()) != int(pack_segment_ids.numel()):
        raise ValueError("Packed FlashAttention sequence lengths must cover the full padded batch.")

    max_length = int(seqlens.max().item())
    return {
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": max_length,
        "max_length_k": max_length,
    }


__all__ = [
    "build_packed_fa2_varlen_kwargs",
    "prepare_packed_model_inputs",
]
