"""Recipe-local FlashAttention 2 patching for packed segment masks.

Qwen3-VL packed training needs two coordinated overrides:
- ``create_causal_mask`` must return the integer segment-id mask directly.
- ``_get_unpad_data`` must interpret segment ids as block-diagonal boundaries.

This matches the working behavior used in the LLaMA-Factory reference. The
previous patch only modified shared masking utils, which left Qwen3-VL's own
causal-mask path inconsistent and produced artificially low packed-sequence
losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import modeling_flash_attention_utils

_patched = False


def apply_packed_fa2_patch() -> None:
    """Permanently monkeypatch Qwen3-VL FA2 packing helpers.

    Call once after model load. No context manager is needed at forward time.
    Calling more than once is a no-op.
    """
    global _patched
    if _patched:
        return

    def _return_segment_mask(*args, **kwargs):
        if "attention_mask" in kwargs:
            return kwargs["attention_mask"]
        if len(args) >= 3:
            return args[2]
        raise ValueError("Unable to recover attention_mask for packed Qwen3-VL FA2 patch.")

    def _get_seqlens_in_batch(attention_mask):
        batch_size = attention_mask.size(0)
        dtype, device = attention_mask.dtype, attention_mask.device
        max_segment_id = torch.max(attention_mask).item()
        counts = torch.zeros((batch_size, max_segment_id), dtype=dtype, device=device)
        for index in range(max_segment_id):
            counts[:, index] = torch.sum(attention_mask == (index + 1), dim=-1)
        counts = counts.flatten()
        return counts[counts.nonzero().squeeze(dim=-1)]

    def _get_unpad_data(attention_mask):
        seqlens_in_batch = _get_seqlens_in_batch(attention_mask)
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch

    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_mod

    qwen3_vl_mod.create_causal_mask = _return_segment_mask
    modeling_flash_attention_utils._get_unpad_data = _get_unpad_data

    _patched = True
