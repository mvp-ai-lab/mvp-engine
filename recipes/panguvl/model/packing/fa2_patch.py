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
        device = attention_mask.device
        # masking_utils._preprocess_mask_arguments casts the 2D mask to bool before
        # returning it from create_causal_mask when FA2 is the backend. If the
        # embedded model is not patched, we receive a bool mask here instead of
        # integer segment IDs. Always use int64 for counts so seq-lengths are
        # preserved correctly (a bool count of 4096 tokens would collapse to 1).
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.long()
        max_segment_id = int(torch.max(attention_mask).item())
        if max_segment_id == 0:
            return torch.zeros(0, dtype=torch.int64, device=device)
        counts = torch.zeros((batch_size, max_segment_id), dtype=torch.int64, device=device)
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

    import sys

    import transformers.masking_utils as masking_utils_mod
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_mod

    # Patch the native Qwen3-VL module (original patch location).
    qwen3_vl_mod.create_causal_mask = _return_segment_mask

    # Patch the source module so any future `from transformers.masking_utils import
    # create_causal_mask` picks up the override.
    masking_utils_mod.create_causal_mask = _return_segment_mask

    # Patch every already-loaded PanguVL trust_remote_code module that imported
    # create_causal_mask at module load time (from X import Y binds the old object).
    # The embedded model is the one that actually calls it per-forward.
    _PANGU_MOD_PREFIX = "transformers_modules.openPangu_hyphen_VL_hyphen_7B"
    for _mod_name, _mod in list(sys.modules.items()):
        if _mod_name.startswith(_PANGU_MOD_PREFIX) and hasattr(_mod, "create_causal_mask"):
            _mod.create_causal_mask = _return_segment_mask

    modeling_flash_attention_utils._get_unpad_data = _get_unpad_data

    _patched = True
