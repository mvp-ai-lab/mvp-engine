"""Recipe-local FlashAttention 2 patching for packed segment masks."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import masking_utils, modeling_flash_attention_utils


@contextmanager
def segmented_flash_attention_patches():
    """Temporarily patch HF FA2 helpers so segment-id masks survive to unpadding."""

    def is_segmented_mask(attention_mask: torch.Tensor | None) -> bool:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            return False
        if torch.is_floating_point(attention_mask) or attention_mask.numel() == 0:
            return False
        return int(attention_mask.max().item()) > 1

    def preprocess_mask_arguments(
        config,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values,
        position_ids: torch.Tensor | None,
        layer_idx: int | None,
    ):
        if not (is_segmented_mask(attention_mask) and config._attn_implementation == "flash_attention_2"):
            return original_preprocess_mask_arguments(
                config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
            )

        if isinstance(attention_mask, (torch.Tensor, masking_utils.BlockMask)) and len(attention_mask.shape) == 4:
            return True, attention_mask, None, None, None

        if config._attn_implementation not in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
            return True, None, None, None, None

        attention_mask = attention_mask.to(device=cache_position.device)
        if past_key_values is not None:
            kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
        else:
            kv_length, kv_offset = attention_mask.shape[-1], 0

        return False, attention_mask, None, kv_length, kv_offset

    def flash_attention_mask(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function=masking_utils.causal_mask_function,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if not is_segmented_mask(attention_mask):
            return original_flash_attention_mask(
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

    def get_unpad_data(attention_mask: torch.Tensor):
        if not is_segmented_mask(attention_mask):
            return original_get_unpad_data(attention_mask)

        indices = torch.nonzero(attention_mask.reshape(-1) != 0, as_tuple=False).flatten()
        segment_lengths: list[int] = []
        for row in attention_mask:
            row = row[row != 0]
            if row.numel() == 0:
                continue
            segment_starts = torch.cat(
                [
                    torch.tensor([0], device=row.device, dtype=torch.long),
                    torch.nonzero(row[1:] != row[:-1], as_tuple=False).flatten() + 1,
                ]
            )
            segment_ends = torch.cat(
                [
                    segment_starts[1:],
                    torch.tensor([row.numel()], device=row.device, dtype=torch.long),
                ]
            )
            segment_lengths.extend((segment_ends - segment_starts).tolist())

        if not segment_lengths:
            raise ValueError("Segmented attention mask must contain at least one non-padding token.")

        seqlens_in_batch = torch.tensor(segment_lengths, dtype=torch.int32, device=attention_mask.device)
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch

    original_preprocess_mask_arguments = masking_utils._preprocess_mask_arguments
    original_flash_attention_mask = masking_utils.flash_attention_mask
    original_get_unpad_data = modeling_flash_attention_utils._get_unpad_data

    with ExitStack() as stack:
        stack.enter_context(patch("transformers.masking_utils._preprocess_mask_arguments", preprocess_mask_arguments))
        stack.enter_context(patch("transformers.masking_utils.flash_attention_mask", flash_attention_mask))
        stack.enter_context(patch("transformers.modeling_flash_attention_utils._get_unpad_data", get_unpad_data))
        yield


__all__ = ["segmented_flash_attention_patches"]
