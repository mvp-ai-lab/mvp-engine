from __future__ import annotations

from pathlib import Path

import pytest
import torch

from recipes.minimal_vlm.dataset.dataset import process_sample


class _FakeProcessor:
    def apply_chat_template(self, conversations, **kwargs):
        messages = conversations[0]
        add_generation_prompt = kwargs.get("add_generation_prompt", False)

        # Simulate tokenized lengths for prefix/upto calls in build_labels.
        if len(messages) == 1 and messages[0]["role"] == "user":
            if add_generation_prompt:
                input_ids = torch.tensor([[11, 12]])
            else:
                input_ids = torch.tensor([[11, 12, 13]])
        else:
            # Full conversation with one assistant turn.
            input_ids = torch.tensor([[21, 22, 23, 24]])

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def _build_sample(tmp_path: Path, messages: list[dict[str, str]]) -> dict[str, object]:
    source_file = tmp_path / "train.jsonl"
    source_file.write_text("{}\n", encoding="utf-8")
    return {
        "__file__": str(source_file),
        "__index_in_file__": 0,
        "messages": messages,
        "images": [],
    }


def test_process_sample_rejects_rows_without_supervised_assistant_tokens(tmp_path: Path) -> None:
    sample = _build_sample(tmp_path, [{"role": "user", "content": "hello"}])

    with pytest.raises(ValueError, match="no supervised assistant tokens"):
        process_sample(sample, processor=_FakeProcessor(), max_length=8)


def test_process_sample_accepts_rows_with_supervised_assistant_tokens(tmp_path: Path) -> None:
    sample = _build_sample(
        tmp_path,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    )

    processed = process_sample(sample, processor=_FakeProcessor(), max_length=8)

    assert torch.any(processed["labels"] != -100)
