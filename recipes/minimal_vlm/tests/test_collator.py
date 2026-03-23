from __future__ import annotations

import torch

from recipes.minimal_vlm.dataset import MinimalVLMCollator


def test_collator_pads_sequences_and_concatenates_multimodal_tensors() -> None:
    collator = MinimalVLMCollator(pad_token_id=0)
    batch = [
        {
            "input_ids": torch.tensor([10, 11, 12, 13]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([-100, -100, 12, 13]),
            "pack_segment_ids": torch.tensor([1, 1, 2, 2]),
            "pixel_values": torch.randn(1, 3, 4, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        {
            "input_ids": torch.tensor([20, 21]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([20, 21]),
            "pack_segment_ids": torch.tensor([1, 1]),
        },
    ]

    model_inputs = collator(batch)

    assert torch.equal(
        model_inputs["input_ids"],
        torch.tensor(
            [
                [10, 11, 12, 13],
                [20, 21, 0, 0],
            ]
        ),
    )
    assert torch.equal(
        model_inputs["attention_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 0, 0],
            ]
        ),
    )
    assert torch.equal(
        model_inputs["labels"],
        torch.tensor(
            [
                [-100, -100, 12, 13],
                [20, 21, -100, -100],
            ]
        ),
    )
    assert torch.equal(
        model_inputs["pack_segment_ids"],
        torch.tensor(
            [
                [1, 1, 2, 2],
                [1, 1, 0, 0],
            ]
        ),
    )
    assert model_inputs["pixel_values"].shape == (1, 3, 4, 4)
    assert torch.equal(model_inputs["image_grid_thw"], torch.tensor([[1, 2, 2]]))
