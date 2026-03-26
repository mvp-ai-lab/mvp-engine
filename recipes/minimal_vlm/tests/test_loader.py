from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from recipes.minimal_vlm.dataset import MinimalVLMCollator, build_dataset


class DummyProcessor:
    def __fingerprint__(self) -> str:
        return "dummy-processor"

    def apply_chat_template(self, conversations, **kwargs):
        messages = conversations[0]
        message_count = len(messages)
        add_generation_prompt = kwargs["add_generation_prompt"]

        if message_count == 1 and add_generation_prompt:
            return {"input_ids": torch.tensor([[20, 21]])}
        if message_count == 1 and not add_generation_prompt:
            return {
                "input_ids": torch.tensor([[20, 21, 22]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        if message_count == 2 and not add_generation_prompt:
            return {
                "input_ids": torch.tensor([[10, 11, 12, 13]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                "pixel_values": torch.randn(1, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        if message_count == 3 and add_generation_prompt:
            return {"input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]])}
        if message_count == 4 and not add_generation_prompt:
            return {
                "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
                "pixel_values": torch.randn(2, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            }

        raise AssertionError(
            f"Unexpected call: message_count={message_count}, add_generation_prompt={add_generation_prompt}"
        )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_torch_loader_batches_rendered_samples(tmp_path: Path) -> None:
    mvp_dataset = pytest.importorskip("mvp_dataset")

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for image_name in ["1.jpg", "2.jpg"]:
        (image_dir / image_name).write_bytes(b"test")

    dataset_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "<image>Who are they?"},
                    {"role": "assistant", "content": "Two players."},
                    {"role": "user", "content": "What are they doing?<image>"},
                    {"role": "assistant", "content": "Celebrating."},
                ],
                "images": ["images/1.jpg", "images/2.jpg"],
            },
            {
                "messages": [{"role": "assistant", "content": "Standalone answer."}],
                "images": [],
            },
        ],
    )

    config = OmegaConf.create(
        {
            "project": {"dir": str(tmp_path / "outputs")},
            "seed": 42,
            "data": {
                "train_path": str(dataset_path),
                "num_workers": 0,
                "jsonl_num_shards": 1,
                "shuffle_buffer": 128,
                "max_seq_len": 16,
            },
        }
    )

    dataset = build_dataset(config, processor=DummyProcessor())
    loader = (
        mvp_dataset.TorchLoader(dataset, num_workers=0)
        .shuffle(buffer_size=4)
        .batch(
            batch_size=2,
            drop_last=True,
            collate_fn=MinimalVLMCollator(pad_token_id=0),
        )
    )

    batch = next(iter(loader))
    assert {"input_ids", "attention_mask", "labels"}.issubset(batch.keys())
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    if "pixel_values" in batch:
        assert "image_grid_thw" in batch
