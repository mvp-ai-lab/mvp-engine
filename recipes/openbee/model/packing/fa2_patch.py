"""Recipe-local FlashAttention 2 patching for packed segment masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import masking_utils, modeling_flash_attention_utils

_patched = False


def _is_segmented_mask(attention_mask: torch.Tensor | None) -> bool:
    """Return True iff attention_mask is a 2D integer segment-id tensor."""
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        return False
    if torch.is_floating_point(attention_mask) or attention_mask.numel() == 0:
        return False
    return int(attention_mask.max().item()) > 1


def apply_packed_fa2_patch() -> None:
    """Permanently monkeypatch HF FA2 shared utils so segment-id masks survive to unpadding.

    Call once after model load. No context manager is needed at forward time.
    Calling more than once is a no-op.

    The patch targets three functions in transformers shared utils — the same code path
    used by all HF FA2 models — so it works across model families without per-model changes.
    """
    global _patched
    if _patched:
        return

    _original_preprocess = masking_utils._preprocess_mask_arguments
    _original_fa2_mask = masking_utils.flash_attention_mask
    _original_get_unpad = modeling_flash_attention_utils._get_unpad_data

    def _preprocess_mask_arguments(
        config,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values,
        position_ids: torch.Tensor | None,
        layer_idx: int | None,
    ):
        if not (_is_segmented_mask(attention_mask) and config._attn_implementation == "flash_attention_2"):
            return _original_preprocess(
                config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
            )

        if config._attn_implementation not in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
            return True, None, None, None, None

        attention_mask = attention_mask.to(device=cache_position.device)
        if past_key_values is not None:
            kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
        else:
            kv_length, kv_offset = attention_mask.shape[-1], 0

        return False, attention_mask, None, kv_length, kv_offset

    def _flash_attention_mask(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function=masking_utils.causal_mask_function,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if not _is_segmented_mask(attention_mask):
            return _original_fa2_mask(
                batch_size=batch_size,
                cache_position=cache_position,
                kv_length=kv_length,
                kv_offset=kv_offset,
                mask_function=mask_function,
                attention_mask=attention_mask,
                **kwargs,
            )

        del batch_size, cache_position, kv_offset, mask_function, kwargs
        return attention_mask[:, -kv_length:]

    # flash_attn's C++ kernel requires max_seqlen to be a concrete Python int even though
    # the declaration says SymInt — the Python binding does not actually support SymInt.
    # Disabling compile here causes a graph break once per forward pass (not per layer).
    @torch.compiler.disable
    def _get_unpad_data(attention_mask: torch.Tensor):
        if not _is_segmented_mask(attention_mask):
            return _original_get_unpad(attention_mask)

        indices = torch.nonzero(attention_mask.reshape(-1) != 0, as_tuple=False).flatten()
        segment_lengths: list[int] = []
        for row in attention_mask:
            row = row[row != 0]
            if row.numel() == 0:
                continue
            # boundaries: positions where the segment id changes, plus start and end sentinels
            boundaries = torch.cat(
                [
                    torch.tensor([0], device=row.device, dtype=torch.long),
                    torch.diff(row).ne(0).nonzero().flatten() + 1,
                    torch.tensor([row.numel()], device=row.device, dtype=torch.long),
                ]
            )
            segment_lengths.extend(torch.diff(boundaries).tolist())

        if not segment_lengths:
            raise ValueError("Segmented attention mask must contain at least one non-padding token.")

        seqlens_in_batch = torch.tensor(segment_lengths, dtype=torch.int32, device=attention_mask.device)
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch

    masking_utils._preprocess_mask_arguments = _preprocess_mask_arguments
    masking_utils.flash_attention_mask = _flash_attention_mask
    modeling_flash_attention_utils._get_unpad_data = _get_unpad_data

    _patched = True
