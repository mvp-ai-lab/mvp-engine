"""Schema handlers for MLLM source-row normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import MLLMMediaSlot, MLLMSegment

ROLE_MAP = {
    "assistant": "assistant",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "tool": "tool",
    "user": "user",
}


class MLLMSchemaHandler:
    """Extension point that normalizes raw source rows into canonical sample fields.

    Subclasses own source-format support. They decide the ordered segment stream,
    explicit loss flags, media slot binding, and any schema metadata.
    """

    def normalize(self, row: Mapping[str, Any]) -> tuple[list[MLLMSegment], list[MLLMMediaSlot], dict[str, Any]]:
        """Normalize one source row.

        Args:
            row: Raw source-row mapping.

        Returns:
            A tuple of ``(segments, media_slots, metadata)``. Segments define the
            ordered text/media stream and loss flags. Media slots describe where
            raw media values live on the source row. Metadata is reserved for
            schema-specific information.
        """
        raise NotImplementedError
