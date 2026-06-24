"""Tokenization handlers for text-only LM sample segments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import LLMSegment

TokenizedFields = tuple[list[int], list[int], list[int]]


@dataclass(slots=True)
class LLMTokenizationHandler:
    """Tokenize normalized text segments and assign labels from segment loss flags."""

    tokenizer: Any
    max_seq_len: int
    ignore_index: int = -100
    add_eos: bool = True

    def __post_init__(self) -> None:
        """Validate tokenization options."""
        if self.max_seq_len <= 0:
            raise ValueError("LLMTokenizationHandler.max_seq_len must be positive.")

    def tokenize(self, segments: Sequence[LLMSegment]) -> list[TokenizedFields]:
        """Tokenize segments into one or more max-length chunks."""
        input_ids: list[int] = []
        labels: list[int] = []
        for segment in segments:
            if segment.type != "text":
                raise ValueError(f"Unsupported LLM segment type: {segment.type!r}.")
            if not isinstance(segment.value, str):
                raise TypeError("LLM text segment value must be a string.")
            segment_ids = self.tokenizer(segment.value, add_special_tokens=False)["input_ids"]
            input_ids.extend(segment_ids)
            labels.extend(segment_ids if segment.loss else [self.ignore_index] * len(segment_ids))

        if self.add_eos and input_ids:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                raise ValueError("Tokenizer has no eos_token_id; required when add_eos=True.")
            eos_loss = segments[-1].loss if segments else True
            input_ids.append(int(eos_token_id))
            labels.append(int(eos_token_id) if eos_loss else self.ignore_index)

        chunks: list[TokenizedFields] = []
        for start in range(0, len(input_ids), self.max_seq_len):
            chunk_input_ids = input_ids[start : start + self.max_seq_len]
            chunk_labels = labels[start : start + self.max_seq_len]
            if chunk_input_ids:
                chunks.append((chunk_input_ids, chunk_labels, [1] * len(chunk_input_ids)))
        return chunks


@dataclass(slots=True)
class LLMPretrainTextTokenizationHandler(LLMTokenizationHandler):
    """Tokenization handler for full-token-loss text pretraining."""
