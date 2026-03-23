from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import masking_utils, modeling_flash_attention_utils

from recipes.minimal_vlm.model.packing.fa2_patch import (
    segmented_flash_attention_patches,
)


def test_segmented_flash_attention_patch_preserves_segment_ids_and_unpad_data() -> None:
    config = SimpleNamespace(_attn_implementation="flash_attention_2")
    attention_mask = torch.tensor([[1, 1, 2, 2, 0], [1, 2, 2, 0, 0]], dtype=torch.long)
    original_preprocess = masking_utils._preprocess_mask_arguments
    original_get_unpad_data = modeling_flash_attention_utils._get_unpad_data

    with segmented_flash_attention_patches():
        _, processed_mask, _, kv_length, kv_offset = masking_utils._preprocess_mask_arguments(
            config,
            input_embeds=torch.randn(2, 5, 8),
            attention_mask=attention_mask,
            cache_position=torch.arange(5),
            past_key_values=None,
            position_ids=None,
            layer_idx=0,
        )
        assert torch.equal(processed_mask, attention_mask)
        assert kv_length == 5
        assert kv_offset == 0

        indices, cu_seqlens, max_seqlen = modeling_flash_attention_utils._get_unpad_data(attention_mask)
        assert torch.equal(indices, torch.tensor([0, 1, 2, 3, 5, 6, 7]))
        assert torch.equal(cu_seqlens, torch.tensor([0, 2, 4, 5, 7], dtype=torch.int32))
        assert max_seqlen == 2

    assert masking_utils._preprocess_mask_arguments is original_preprocess
    assert modeling_flash_attention_utils._get_unpad_data is original_get_unpad_data
