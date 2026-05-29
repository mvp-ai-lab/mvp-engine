"""Recipe-local FlashAttention 2 patching for Qwen3 packed segment masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import modeling_flash_attention_utils

_patched = False


def apply_packed_fa2_patch() -> None:
    """Permanently monkeypatch Qwen3/Qwen3-MoE FA2 packing helpers."""
    global _patched
    if _patched:
        return

    def _return_segment_mask(*args, **kwargs):
        """Return Qwen3's integer packed segment mask unchanged."""
        if "attention_mask" in kwargs:
            return kwargs["attention_mask"]
        if len(args) >= 3:
            return args[2]
        raise ValueError("Unable to recover attention_mask for packed Qwen3 FA2 patch.")

    def _get_seqlens_in_batch(attention_mask):
        """Convert packed segment ids to FA2 per-segment sequence lengths."""
        batch_size = attention_mask.size(0)
        device = attention_mask.device
        max_segment_id = int(torch.max(attention_mask).item())
        if max_segment_id <= 0:
            raise ValueError("Packed Qwen3 FA2 attention_mask must contain at least one valid token.")

        counts = torch.zeros((batch_size, max_segment_id), dtype=torch.int32, device=device)
        for index in range(max_segment_id):
            counts[:, index] = torch.sum(attention_mask == (index + 1), dim=-1)
        counts = counts.flatten()
        return counts[counts.nonzero().squeeze(dim=-1)]

    def _get_unpad_data(attention_mask):
        """Build FA2 unpadding metadata from packed segment-id masks."""
        seqlens_in_batch = _get_seqlens_in_batch(attention_mask)
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch

    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

    qwen3_mod.create_causal_mask = _return_segment_mask
    qwen3_mod.create_sliding_window_causal_mask = _return_segment_mask
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_mod

        qwen3_moe_mod.create_causal_mask = _return_segment_mask
        qwen3_moe_mod.create_sliding_window_causal_mask = _return_segment_mask
    except ImportError:
        pass
    modeling_flash_attention_utils._get_unpad_data = _get_unpad_data

    _patched = True
