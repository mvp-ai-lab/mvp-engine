"""Prepare packed text batches for the Qwen3 forward pass (SDPA path)."""

import torch

from mvp_engine.kit.llm.data import build_packed_block_causal_mask


def prepare_packed_model_inputs(batch: dict, *, mask_dtype: torch.dtype) -> dict:
    """Convert packed token metadata into Qwen3 text model inputs."""
    pack_segment_ids = batch.get("pack_segment_ids")
    if pack_segment_ids is None:
        raise ValueError("Packed batches must include pack_segment_ids.")

    # Drop bookkeeping fields the model forward does not accept.
    batch.pop("source_sample_num", None)
    batch.pop("num_input_tokens", None)
    batch.pop("num_loss_tokens", None)
    batch.pop("num_source_samples", None)

    batch["position_ids"] = build_packed_text_position_ids(batch["input_ids"], pack_segment_ids)
    # A 4D block-causal mask isolates each packed document (SDPA / eager path).
    batch["attention_mask"] = build_packed_block_causal_mask(pack_segment_ids, dtype=mask_dtype)
    return batch


def build_packed_text_position_ids(input_ids: torch.Tensor, pack_segment_ids: torch.Tensor) -> torch.Tensor:
    """Reset position ids to 0 at the start of each packed document."""
    position_ids = torch.zeros_like(input_ids)
    for row in range(input_ids.size(0)):
        segment_ids = pack_segment_ids[row]
        for segment_id in segment_ids[segment_ids > 0].unique():
            mask = segment_ids == segment_id
            length = int(mask.sum())
            position_ids[row, mask] = torch.arange(length, device=input_ids.device)
    return position_ids
