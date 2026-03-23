from __future__ import annotations

from types import SimpleNamespace

import torch

from recipes.minimal_vlm.model.packing.qwen3_vl import (
    build_qwen3_vl_packed_position_ids,
)


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        image_token_id=99,
        video_token_id=97,
        vision_start_token_id=98,
        vision_config=SimpleNamespace(spatial_merge_size=1),
    )


def test_build_qwen3_vl_packed_position_ids_resets_text_positions_per_segment() -> None:
    position_ids = build_qwen3_vl_packed_position_ids(
        input_ids=torch.tensor([[10, 11, 20, 21, 22]]),
        pack_segment_ids=torch.tensor([[1, 1, 2, 2, 2]]),
        image_grid_thw=None,
        model_config=_model_config(),
    )

    expected = torch.tensor(
        [
            [[0, 1, 0, 1, 2]],
            [[0, 1, 0, 1, 2]],
            [[0, 1, 0, 1, 2]],
        ]
    )
    assert torch.equal(position_ids, expected)


def test_build_qwen3_vl_packed_position_ids_handles_multimodal_segments() -> None:
    position_ids = build_qwen3_vl_packed_position_ids(
        input_ids=torch.tensor([[10, 11, 98, 99, 99, 99, 99, 20]]),
        pack_segment_ids=torch.tensor([[1, 1, 2, 2, 2, 2, 2, 2]]),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        model_config=_model_config(),
    )

    expected = torch.tensor(
        [
            [[0, 1, 0, 1, 1, 1, 1, 3]],
            [[0, 1, 0, 1, 1, 2, 2, 3]],
            [[0, 1, 0, 1, 2, 1, 2, 3]],
        ]
    )
    assert torch.equal(position_ids, expected)
