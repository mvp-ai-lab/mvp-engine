"""Prepare packed text batches for the Qwen3 forward pass."""

import torch

from mvp_engine.kit.llm.data import build_packed_block_causal_mask


def prepare_packed_model_inputs(
    batch: dict,
    *,
    attn_implementation: str | None,
    mask_dtype: torch.dtype,
) -> dict:
    """Convert packed token metadata into Qwen3 text model inputs."""
    pack_segment_ids = batch.get("pack_segment_ids")
    if pack_segment_ids is None:
        raise ValueError("Packed batches must include pack_segment_ids.")

    # Drop bookkeeping fields the model forward does not accept.
    batch.pop("source_sample_num", None)
    batch.pop("num_input_tokens", None)
    batch.pop("num_loss_tokens", None)
    batch.pop("num_source_samples", None)

    if "position_ids" not in batch:
        batch["position_ids"] = build_packed_text_position_ids(batch["input_ids"], pack_segment_ids)
    if attn_implementation == "flash_attention_2":
        batch["attention_mask"] = None
        batch.update(build_packed_fa2_varlen_kwargs(pack_segment_ids))
    else:
        batch["attention_mask"] = build_packed_block_causal_mask(pack_segment_ids, dtype=mask_dtype)
    return batch


def build_packed_text_position_ids(input_ids: torch.Tensor, pack_segment_ids: torch.Tensor) -> torch.Tensor:
    """Reset position ids to 0 at the start of each packed stream segment."""
    position_ids = torch.zeros_like(input_ids)
    for row in range(input_ids.size(0)):
        segment_ids = pack_segment_ids[row]
        for segment_id in segment_ids[segment_ids > 0].unique():
            mask = segment_ids == segment_id
            length = int(mask.sum())
            position_ids[row, mask] = torch.arange(length, device=input_ids.device)
    return position_ids


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
