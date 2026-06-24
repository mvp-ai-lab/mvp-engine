"""Schema handlers for text-only LM source-row normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .types import LLMSegment


class LLMSchemaHandler:
    """Extension point that normalizes raw source rows into canonical text segments."""

    def check_row(self, row: Mapping[str, Any]) -> str | None:
        """Return a compact invalid reason, or ``None`` when the row can be normalized."""
        del row
        return None

    def normalize(self, row: Mapping[str, Any]) -> tuple[list[LLMSegment], dict[str, Any]]:
        """Normalize one source row into ordered text segments and metadata."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class LLMPretrainTextSchemaHandler(LLMSchemaHandler):
    """Normalize a pretraining row that stores one document in a text field."""

    text_field: str = "data"
    loss: bool = True

    def check_row(self, row: Mapping[str, Any]) -> str | None:
        """Validate the configured text field before expensive tokenization."""
        text = row.get(self.text_field)
        if not isinstance(text, str) or not text.strip():
            return "raw.invalid_text"
        return None

    def normalize(self, row: Mapping[str, Any]) -> tuple[list[LLMSegment], dict[str, Any]]:
        """Return the text document as one supervised segment."""
        reason = self.check_row(row)
        if reason is not None:
            raise ValueError(reason)
        return [LLMSegment(type="text", loss=self.loss, value=row[self.text_field])], {"text_field": self.text_field}
