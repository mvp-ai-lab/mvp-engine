"""Media rendering, loading, merging, and collation hooks for MLLM data pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .types import MLLMMediaSlot, MLLMSegment


def empty_model_sample() -> dict[str, torch.Tensor]:
    """Build an empty model-input sentinel for invalid samples.

    Returns:
        Empty ``input_ids``, ``attention_mask``, and ``labels`` tensors used by guards
        and packing code to drop unusable samples.
    """
    return {
        "input_ids": torch.empty(0, dtype=torch.long),
        "attention_mask": torch.empty(0, dtype=torch.long),
        "labels": torch.empty(0, dtype=torch.long),
    }


@dataclass(frozen=True, slots=True)
class RenderedMedia:
    """Rendered placeholder text for one media slot.

    Attributes:
        media_id: Stable id that matches the source ``MLLMMediaSlot``.
        media_type: Media type handled by a registered media type handler.
        text: Placeholder text inserted into the rendered token stream.
        metadata: Optional model-specific metadata produced while rendering.
    """

    media_id: str
    media_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MLLMMediaTypeHandler:
    """Extension point for one media type.

    A media type handler owns the model-specific behavior for a single modality,
    such as image or video. It renders placeholder text before tokenization, loads
    raw media values into model tensors after references are resolved, and merges
    those tensors at pack and batch boundaries.
    """

    media_type: str

    def render(
        self,
        slot: MLLMMediaSlot,
        *,
        processor: Any,
        tokenizer: Any,
    ) -> RenderedMedia:
        """Render one media slot into placeholder text.

        Args:
            slot: Normalized media slot produced by the schema handler.
            processor: Model processor attached to the parent media handler.
            tokenizer: Tokenizer exposed by the processor.

        Returns:
            Rendered placeholder text and metadata for the slot.
        """
        raise NotImplementedError

    def default_token(self, processor: Any) -> str:
        """Return this media type's default token text.

        Args:
            processor: Model processor attached to the parent media handler.

        Returns:
            Default placeholder token text, or an empty string when the media type
            has no generic placeholder token.
        """
        return ""

    def placeholder_aliases(self, processor: Any) -> tuple[str, ...]:
        """Return source placeholder aliases accepted by this handler.

        Args:
            processor: Model processor attached to the parent media handler.

        Returns:
            Placeholder strings that schema handlers may recognize for this media type.
        """
        token = self.default_token(processor)
        return (token,) if token else ()

    def load(
        self,
        slots: Sequence[MLLMMediaSlot],
        values: Sequence[Any],
        *,
        processor: Any,
    ) -> dict[str, Any]:
        """Load media values into model-input media fields.

        Args:
            slots: Media slots of this handler's type.
            values: Current raw media values read from the sample.
            processor: Model processor attached to the parent media handler.

        Returns:
            Model-input media fields. Returning an empty model-input sentinel marks
            the whole sample unusable.
        """
        del slots, values, processor
        return {}

    def merge_pack(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Merge this media type across samples in one pack.

        Args:
            samples: Per-source-sample model-input dictionaries in one pack.

        Returns:
            Media fields for the finalized packed sample.
        """
        return {}

    def collate(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Collate this media type across model-input samples.

        Args:
            batch: Packed model-input dictionaries in one batch.

        Returns:
            Batched media fields.
        """
        return {}


@dataclass(slots=True)
class MLLMMediaHandler:
    """Dispatch media lifecycle behavior to registered media-type handlers.

    Attributes:
        processor: Model processor shared by all media type handlers.
        handlers: Mapping from media type name to media type handler.
    """

    processor: Any
    handlers: Mapping[str, MLLMMediaTypeHandler] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize handler storage."""
        self.handlers = dict(self.handlers)

    def render(
        self,
        segments: Sequence[MLLMSegment],
        media_slots: Sequence[MLLMMediaSlot],
    ) -> list[MLLMSegment]:
        """Render media segment ids into model token text.

        Args:
            segments: Normalized text/media segments. Text segments already contain text;
                media segments contain media ids.
            media_slots: Normalized media slots available to the sample.

        Returns:
            Segments whose values are all strings ready for tokenization.

        Raises:
            ValueError: If the processor lacks a tokenizer, a media type is unsupported,
                or segment and slot media types disagree.
            TypeError: If a segment value has an invalid type.
        """
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Processor must expose a tokenizer for media rendering.")

        rendered_media = self._render_media_placeholders(media_slots, tokenizer=tokenizer)
        rendered_segments = []
        for segment in segments:
            if segment.type == "text":
                if not isinstance(segment.value, str):
                    raise TypeError("text segment value must be a string.")
                rendered_segments.append(segment)
                continue

            if not isinstance(segment.value, str):
                raise TypeError("media segment value must be a media id string.")
            item = rendered_media[segment.value]
            if item.media_type != segment.type:
                raise ValueError("media segment type does not match prepared media type.")
            rendered_segments.append(MLLMSegment(type=segment.type, loss=segment.loss, value=item.text))

        return rendered_segments

    def load(self, sample: object, media_slots: Sequence[MLLMMediaSlot]) -> dict[str, Any]:
        """Load model media fields from the sample's current raw media values.

        Args:
            sample: Sample-like object exposing raw media fields as attributes.
            media_slots: Media slots that should be loaded.

        Returns:
            Model-input media fields merged across all registered media types.
        """
        model_media: dict[str, Any] = {}
        for media_type, handler in self.handlers.items():
            typed_slots = [slot for slot in media_slots if slot.media_type == media_type]
            if not typed_slots:
                continue
            typed_values = [self._slot_value(sample, slot) for slot in typed_slots]
            loaded = handler.load(typed_slots, typed_values, processor=self.processor)
            if _is_empty_model_sample(loaded):
                return loaded
            model_media.update(loaded)
        return model_media

    def merge_pack(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Merge media fields across samples in one pack.

        Args:
            samples: Per-source-sample model-input dictionaries in one pack.

        Returns:
            Media fields for the finalized packed sample.
        """
        merged: dict[str, Any] = {}
        for handler in self.handlers.values():
            merged.update(handler.merge_pack(samples))
        return merged

    def collate(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Collate all registered media fields.

        Args:
            batch: Packed model-input dictionaries in one batch.

        Returns:
            Batched media fields merged across all registered media types.
        """
        collated: dict[str, Any] = {}
        for handler in self.handlers.values():
            collated.update(handler.collate(batch))
        return collated

    def _render_media_placeholders(
        self,
        media_slots: Sequence[MLLMMediaSlot],
        *,
        tokenizer: Any,
    ) -> dict[str, RenderedMedia]:
        """Render every media slot into placeholder text keyed by media id."""
        rendered: dict[str, RenderedMedia] = {}
        for slot in media_slots:
            handler = self.handlers.get(slot.media_type)
            if handler is None:
                raise ValueError(f"Unsupported media type {slot.media_type!r}; no handler is registered.")
            rendered[slot.media_id] = handler.render(slot, processor=self.processor, tokenizer=tokenizer)
        return rendered

    @staticmethod
    def _slot_value(sample: object, slot: MLLMMediaSlot) -> Any:
        """Read the current raw media value for one media slot."""
        value = getattr(sample, slot.field)
        if slot.index is not None:
            value = value[slot.index]
        if isinstance(value, dict):
            return value.get("value", value.get(slot.media_type, value))
        return value


def _is_empty_model_sample(sample: dict[str, Any]) -> bool:
    """Return whether media loading produced an empty-sample sentinel."""
    input_ids = sample.get("input_ids")
    return isinstance(input_ids, torch.Tensor) and input_ids.numel() == 0
