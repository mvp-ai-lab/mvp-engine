"""Collator tests for Video VLM multimodal batches."""

import torch

from recipes.video_vlm.dataset.collator import VideoVLMCollator


class DummyProcessor:
    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("video-only batches must not request dummy image inputs")


def test_video_only_batch_does_not_append_dummy_image():
    collator = VideoVLMCollator(pad_token_id=0, processor=DummyProcessor())
    sample = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([-100, -100, 3], dtype=torch.long),
        "pixel_values_videos": torch.zeros(1, 3, 8, 448, 448),
        "video_grid_thw": torch.tensor([[8, 32, 32]], dtype=torch.long),
        "patch_positions": torch.zeros(1024, 3, dtype=torch.long),
    }

    batch = collator([sample])

    assert "pixel_values" not in batch
    assert "image_grid_thw" not in batch
    assert batch["pixel_values_videos"].shape == (1, 3, 8, 448, 448)
    assert batch["video_grid_thw"].tolist() == [[8, 32, 32]]
    assert batch["patch_positions"].shape == (1024, 3)
