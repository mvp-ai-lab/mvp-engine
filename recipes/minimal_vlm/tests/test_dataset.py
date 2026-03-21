from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from recipes.minimal_vlm.dataset import build_dataset, process_sample


class DummyProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
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
        if message_count == 3 and not add_generation_prompt:
            return {
                "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
                "pixel_values": torch.randn(2, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            }
        if message_count == 3 and add_generation_prompt:
            return {"input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]])}
        if message_count == 4 and not add_generation_prompt:
            return {
                "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
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


def _annotate_sample(record: dict, dataset_path: Path, *, index_in_file: int = 0) -> dict:
    return {
        "__file__": str(dataset_path.resolve()),
        "__index_in_file__": index_in_file,
        "__key__": f"{dataset_path.resolve()}:{index_in_file}",
        **record,
    }


def test_process_sample_resolves_relative_image_paths_and_renders_messages(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for image_name in ["1.jpg", "2.jpg"]:
        (image_dir / image_name).write_bytes(b"test")

    dataset_path = tmp_path / "demo.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "<image>Who are they?"},
            {"role": "assistant", "content": "Two players."},
            {"role": "user", "content": "What are they doing?<image>"},
        ],
        "images": ["images/1.jpg", "images/2.jpg"],
    }

    processor = DummyProcessor()
    sample = process_sample(_annotate_sample(record, dataset_path), processor=processor, max_length=16)

    full_conversation = next(
        conversations[0]
        for conversations, kwargs in processor.calls
        if len(conversations[0]) == 3 and not kwargs["add_generation_prompt"]
    )
    first_message = full_conversation[0]
    assert first_message["content"][0]["type"] == "image"
    assert first_message["content"][0]["image"] == str((image_dir / "1.jpg").resolve())
    assert first_message["content"][1] == {"type": "text", "text": "Who are they?"}

    third_message = full_conversation[2]
    assert third_message["content"][0] == {"type": "text", "text": "What are they doing?"}
    assert third_message["content"][1]["type"] == "image"
    assert third_message["content"][1]["image"] == str((image_dir / "2.jpg").resolve())
    assert torch.equal(sample["input_ids"], torch.tensor([10, 11, 12, 13, 14, 15]))
    assert torch.equal(sample["labels"], torch.tensor([-100, -100, 12, 13, -100, -100]))


def test_process_sample_rejects_placeholder_and_image_count_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "1.jpg").write_bytes(b"test")

    dataset_path = tmp_path / "demo.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "<image><image>Describe this."},
            {"role": "assistant", "content": "Example."},
        ],
        "images": ["images/1.jpg"],
    }

    with pytest.raises(ValueError, match="image placeholders"):
        process_sample(_annotate_sample(record, dataset_path), processor=DummyProcessor(), max_length=16)


def test_build_dataset_returns_rendered_mvp_dataset_samples(tmp_path: Path) -> None:
    pytest.importorskip("mvp_dataset")

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "1.jpg").write_bytes(b"test")

    dataset_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "<image>Who is this?"},
                    {"role": "assistant", "content": "An example."},
                ],
                "images": ["images/1.jpg"],
            }
        ],
    )

    config = OmegaConf.create(
        {
            "project": {"dir": str(tmp_path / "outputs"), "seed": 42},
            "data": {
                "train_path": str(dataset_path),
                "num_workers": 0,
                "jsonl_num_shards": 1,
                "max_seq_len": 16,
            },
        }
    )

    processor = DummyProcessor()
    dataset = build_dataset(config, processor=processor)
    sample = next(iter(dataset))
    shard_dir = dataset_path.parent / ".jsonl_shards"

    full_conversation = next(
        conversations[0]
        for conversations, kwargs in processor.calls
        if len(conversations[0]) == 2 and not kwargs["add_generation_prompt"]
    )
    assert full_conversation[0]["content"][0]["type"] == "image"
    assert full_conversation[0]["content"][1] == {"type": "text", "text": "Who is this?"}
    assert full_conversation[1]["content"] == [{"type": "text", "text": "An example."}]
    assert torch.equal(sample["input_ids"], torch.tensor([10, 11, 12, 13]))
    assert torch.equal(sample["labels"], torch.tensor([-100, -100, 12, 13]))
    assert shard_dir.is_dir()
