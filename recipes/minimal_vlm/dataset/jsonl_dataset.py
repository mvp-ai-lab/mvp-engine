"""JSONL dataset utilities for the minimal VLM recipe."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

IMAGE_PLACEHOLDER = "<image>"


class MinimalVlmJsonlDataset(Dataset[dict[str, Any]]):
    """Load multi-turn image-text conversations from a local JSONL file."""

    def __init__(self, jsonl_path: str | Path, *, image_placeholder: str = IMAGE_PLACEHOLDER) -> None:
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.image_placeholder = image_placeholder

        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.jsonl_path}")

        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError(f"No valid samples found in dataset: {self.jsonl_path}")

    def _load_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                samples.append(self._parse_record(record, line_number))
        return samples

    def _parse_record(self, record: dict[str, Any], line_number: int) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError(f"{self.jsonl_path}:{line_number} must contain a JSON object.")

        messages = record.get("messages")
        images = record.get("images", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{self.jsonl_path}:{line_number} has invalid `messages`.")
        if not isinstance(images, list):
            raise ValueError(f"{self.jsonl_path}:{line_number} has invalid `images`.")

        resolved_images = [self._resolve_image_path(image_path, line_number) for image_path in images]
        placeholder_count = sum(self._count_placeholders(message) for message in messages)
        if placeholder_count != len(resolved_images):
            raise ValueError(
                f"{self.jsonl_path}:{line_number} has {placeholder_count} image placeholders but "
                f"{len(resolved_images)} image paths."
            )

        image_iter = iter(resolved_images)
        rendered_messages = [self._render_message(message, image_iter, line_number) for message in messages]

        try:
            next(image_iter)
        except StopIteration:
            pass
        else:
            raise ValueError(f"{self.jsonl_path}:{line_number} contains unused image paths.")

        return {
            "messages": rendered_messages,
            "image_paths": [str(path) for path in resolved_images],
            "line_number": line_number,
        }

    def _resolve_image_path(self, image_path: Any, line_number: int) -> Path:
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"{self.jsonl_path}:{line_number} contains an invalid image path: {image_path!r}")

        resolved_path = (self.jsonl_path.parent / image_path).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"{self.jsonl_path}:{line_number} references missing image: {resolved_path}")
        return resolved_path

    def _count_placeholders(self, message: dict[str, Any]) -> int:
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Each message must contain string `content`.")
        return content.count(self.image_placeholder)

    def _render_message(
        self,
        message: dict[str, Any],
        image_iter: Iterable[Path],
        line_number: int,
    ) -> dict[str, Any]:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{self.jsonl_path}:{line_number} contains an invalid role: {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"{self.jsonl_path}:{line_number} contains non-string content.")

        content_blocks: list[dict[str, str]] = []
        text_segments = content.split(self.image_placeholder)
        for index, segment in enumerate(text_segments):
            if segment:
                content_blocks.append({"type": "text", "text": segment})
            if index < len(text_segments) - 1:
                try:
                    image_path = next(image_iter)
                except StopIteration as exc:
                    raise ValueError(f"{self.jsonl_path}:{line_number} is missing an image for a placeholder.") from exc
                content_blocks.append({"type": "image", "image": str(image_path)})

        return {
            "role": role,
            "content": content_blocks,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def build_dataset(config: Any) -> MinimalVlmJsonlDataset:
    """Build the training dataset for the minimal VLM recipe."""
    dataset_path = config.data.train_path
    if dataset_path is None:
        raise ValueError("Missing `data.train_path` for the minimal VLM recipe.")

    return MinimalVlmJsonlDataset(dataset_path)
