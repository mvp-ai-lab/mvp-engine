from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes.minimal_vlm.dataset import MinimalVlmJsonlDataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_dataset_resolves_relative_image_paths_and_renders_messages(
    tmp_path: Path,
) -> None:
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
                ],
                "images": ["images/1.jpg", "images/2.jpg"],
            }
        ],
    )

    dataset = MinimalVlmJsonlDataset(dataset_path)
    sample = dataset[0]

    first_message = sample["messages"][0]
    assert first_message["content"][0]["type"] == "image"
    assert first_message["content"][0]["image"] == str((image_dir / "1.jpg").resolve())
    assert first_message["content"][1] == {"type": "text", "text": "Who are they?"}

    third_message = sample["messages"][2]
    assert third_message["content"][0] == {
        "type": "text",
        "text": "What are they doing?",
    }
    assert third_message["content"][1]["type"] == "image"
    assert third_message["content"][1]["image"] == str((image_dir / "2.jpg").resolve())


def test_dataset_rejects_placeholder_and_image_count_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "1.jpg").write_bytes(b"test")

    dataset_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "<image><image>Describe this."},
                    {"role": "assistant", "content": "Example."},
                ],
                "images": ["images/1.jpg"],
            }
        ],
    )

    with pytest.raises(ValueError, match="image placeholders"):
        MinimalVlmJsonlDataset(dataset_path)
