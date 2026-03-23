from __future__ import annotations

import torch

from recipes.minimal_vlm.model.packing.masks import build_packed_block_causal_mask


def test_build_packed_block_causal_mask_isolates_segments() -> None:
    attention_mask = build_packed_block_causal_mask(
        torch.tensor([[1, 1, 2, 2, 0]]),
        dtype=torch.float32,
    )

    allowed = attention_mask[0, 0].eq(0)
    assert bool(allowed[0, 0])
    assert bool(allowed[1, 0])
    assert not bool(allowed[1, 2])
    assert bool(allowed[3, 2])
    assert not bool(allowed[3, 1])
    assert not bool(allowed[4].any())
