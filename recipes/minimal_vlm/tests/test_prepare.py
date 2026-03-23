from __future__ import annotations

from types import SimpleNamespace

import torch

from recipes.minimal_vlm.model.packing.prepare import prepare_packed_model_inputs


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        image_token_id=99,
        video_token_id=97,
        vision_start_token_id=98,
        vision_config=SimpleNamespace(spatial_merge_size=1),
    )


def test_prepare_packed_model_inputs_uses_block_mask_for_non_fa2() -> None:
    batch = {
        "input_ids": torch.tensor([[10, 11, 20, 21]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": torch.tensor([[10, 11, 20, 21]]),
        "pack_segment_ids": torch.tensor([[1, 1, 2, 2]]),
    }

    prepared = prepare_packed_model_inputs(
        batch,
        model_config=_model_config(),
        attn_implementation="sdpa",
        mask_dtype=torch.float32,
    )

    assert "pack_segment_ids" not in prepared
    assert prepared["attention_mask"].shape == (1, 1, 4, 4)
    assert prepared["position_ids"].shape == (3, 1, 4)


def test_prepare_packed_model_inputs_uses_segment_ids_for_fa2() -> None:
    batch = {
        "input_ids": torch.tensor([[10, 11, 20, 21]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": torch.tensor([[10, 11, 20, 21]]),
        "pack_segment_ids": torch.tensor([[1, 1, 2, 2]]),
    }

    prepared = prepare_packed_model_inputs(
        batch,
        model_config=_model_config(),
        attn_implementation="flash_attention_2",
        mask_dtype=torch.float32,
    )

    assert "pack_segment_ids" not in prepared
    assert torch.equal(prepared["attention_mask"], torch.tensor([[1, 1, 2, 2]]))
    assert prepared["position_ids"].shape == (3, 1, 4)
