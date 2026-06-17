"""Tokenization handlers for rendered MLLM sample segments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import MLLMSegment


@dataclass(slots=True)
class MLLMTokenizationHandler:
    """Tokenize rendered segments and assign labels according to each segment's loss flag.

    Attributes:
        processor: Model processor exposing a tokenizer.
        max_seq_len: Maximum number of tokens retained from one sample.
        ignore_index: Label value assigned to segments whose ``loss`` flag is false.
    """

    processor: Any
    max_seq_len: int
    ignore_index: int = -100

    def __post_init__(self) -> None:
        """Validate tokenization options.

        Raises:
            ValueError: If ``max_seq_len`` is not positive.
        """
        if self.max_seq_len <= 0:
            raise ValueError("MLLMTokenizationHandler.max_seq_len must be positive.")

    def tokenize(
        self,
        rendered_segments: Sequence[MLLMSegment],
    ) -> tuple[list[int], list[int], list[int]]:
        """Tokenize rendered sample segments into input ids, labels, and attention mask.

        Args:
            rendered_segments: Ordered segments whose values are already rendered as strings.

        Returns:
            ``(input_ids, labels, attention_mask)`` lists aligned token by token.

        Raises:
            ValueError: If the processor has no tokenizer or truncation would cut a media segment.
            TypeError: If a rendered segment value is not a string.
        """
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Processor must expose a tokenizer for MLLM tokenization.")

        input_ids: list[int] = []
        labels: list[int] = []
        for segment in rendered_segments:
            if len(input_ids) >= self.max_seq_len:
                break
            if not isinstance(segment.value, str):
                raise TypeError("rendered segment value must be a string.")

            segment_ids = tokenizer(segment.value, add_special_tokens=False)["input_ids"]
            keep_len = min(len(segment_ids), self.max_seq_len - len(input_ids))
            if segment.type != "text" and keep_len < len(segment_ids):
                raise ValueError("truncation would cut media tokens.")

            kept_ids = segment_ids[:keep_len]
            input_ids.extend(kept_ids)
            labels.extend(kept_ids if segment.loss else [self.ignore_index] * len(kept_ids))

        return input_ids, labels, [1] * len(input_ids)
