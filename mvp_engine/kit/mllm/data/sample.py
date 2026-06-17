"""Stateful sample and packed-sample objects for MLLM data pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from mvp_engine.utils.log import simple_info

from .media import MLLMMediaHandler, empty_model_sample
from .schema import MLLMSchemaHandler
from .spec import MLLMSampleSpec
from .tokenization import MLLMTokenizationHandler
from .types import MLLMMediaSlot, MLLMSegment


@dataclass(slots=True, init=False)
class MLLMSample:
    """Own the lifecycle state for one source row.

    A sample starts with raw source fields. Tokenization lazily fills normalized
    segments and token fields, while media loading later fills model-specific media
    tensors after references have been resolved by the dataset pipeline.
    """

    _raw: dict[str, Any]
    schema_handler: MLLMSchemaHandler
    media_handler: MLLMMediaHandler
    tokenization_handler: MLLMTokenizationHandler
    _segments: list[MLLMSegment] | None = field(default=None, init=False, repr=False)
    _media_slots: list[MLLMMediaSlot] = field(default_factory=list, init=False, repr=False)
    _metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _input_ids: list[int] | None = field(default=None, init=False, repr=False)
    _labels: list[int] | None = field(default=None, init=False, repr=False)
    _attention_mask: list[int] | None = field(default=None, init=False, repr=False)
    _model_media: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __init__(
        self,
        raw: Mapping[str, Any],
        *,
        schema_handler: MLLMSchemaHandler,
        media_handler: MLLMMediaHandler,
        tokenization_handler: MLLMTokenizationHandler,
    ) -> None:
        """Store raw row and handlers; derived sample state is filled lazily.

        Args:
            raw: Source row mapping.
            schema_handler: Handler that converts the row into loss-marked segments and media slots.
            media_handler: Handler that renders media placeholders and loads model media tensors.
            tokenization_handler: Handler that converts rendered segments into token fields.
        """
        self._raw = dict(raw)
        self.schema_handler = schema_handler
        self.media_handler = media_handler
        self.tokenization_handler = tokenization_handler
        self._segments = None
        self._media_slots = []
        self._metadata = {}
        self._input_ids = None
        self._labels = None
        self._attention_mask = None
        self._model_media = {}

    def __getattr__(self, name: str) -> Any:
        """Read source-row fields as sample attributes.

        Args:
            name: Source-row field name.

        Returns:
            The raw source-row value.

        Raises:
            AttributeError: If the raw row does not contain ``name``.
        """
        try:
            raw = object.__getattribute__(self, "_raw")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        try:
            return raw[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        """Write source-row fields as sample attributes.

        Args:
            name: Attribute or source-row field name.
            value: Value to store.
        """
        if name.startswith("_") or name in {"schema_handler", "media_handler", "tokenization_handler"}:
            object.__setattr__(self, name, value)
            return

        raw = object.__getattribute__(self, "_raw")
        raw[name] = value

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *, sample_spec: MLLMSampleSpec) -> MLLMSample:
        """Build a lazy sample from one raw source row and sample spec.

        Args:
            row: Raw row yielded by the source dataset.
            sample_spec: Per-model sample handlers.

        Returns:
            A sample with no derived state materialized yet.
        """
        return cls(
            raw=dict(row),
            schema_handler=sample_spec.schema_handler,
            media_handler=sample_spec.media_handler,
            tokenization_handler=sample_spec.tokenization_handler,
        )

    @property
    def input_ids(self) -> list[int]:
        """Token ids, tokenized lazily on first access.

        Returns:
            Token ids, or an empty list if the sample was marked invalid.
        """
        self.tokenize()
        return self._input_ids or []

    @property
    def labels(self) -> list[int]:
        """Training labels, tokenized lazily on first access.

        Returns:
            Label ids with ignored positions set to the tokenization handler's ``ignore_index``.
        """
        self.tokenize()
        return self._labels or []

    @property
    def attention_mask(self) -> list[int]:
        """Attention mask, tokenized lazily on first access.

        Returns:
            A 1D attention mask aligned with ``input_ids``.
        """
        self.tokenize()
        return self._attention_mask or []

    @property
    def token_length(self) -> int:
        """Return the lazy token sequence length.

        Returns:
            Number of materialized input tokens.
        """
        return len(self.input_ids)

    def tokenize(self) -> MLLMSample:
        """Normalize the row, render media placeholders, and materialize token fields once.

        Returns:
            ``self`` for pipeline chaining.

        Notes:
            Invalid rows are logged at debug level and converted to an empty sample
            sentinel so downstream guards can drop them consistently.
        """
        if self._input_ids is not None:
            return self

        try:
            if self._segments is None:
                self._segments, self._media_slots, self._metadata = self.schema_handler.normalize(self._raw)
            rendered_segments = self.media_handler.render(
                self._segments,
                self._media_slots,
            )
            self._input_ids, self._labels, self._attention_mask = self.tokenization_handler.tokenize(
                rendered_segments,
            )
            if not self._input_ids:
                raise ValueError("sample has no tokens after tokenization/truncation.")
            if not any(label != self.tokenization_handler.ignore_index for label in self.labels):
                raise ValueError("sample has no supervised tokens after tokenization/truncation.")
            self._model_media = {}
        except Exception as exc:
            simple_info(exc, level="debug")
            self.set_empty()
        return self

    def load_media(self) -> MLLMSample:
        """Load model media fields from the sample's current raw media values.

        Returns:
            ``self`` with ``_model_media`` filled, or marked empty when media loading
            determines the sample is unusable.
        """
        if self.token_length > 0:
            model_media = self.media_handler.load(self, self._media_slots)
            input_ids = model_media.get("input_ids")
            if isinstance(input_ids, torch.Tensor) and not input_ids.numel():
                self.set_empty()
            else:
                self._model_media = model_media
        return self

    def to_model_inputs(self) -> dict[str, Any]:
        """Return this sample as token tensors plus any loaded model media fields.

        Returns:
            A model-input dictionary containing ``input_ids``, ``attention_mask``,
            ``labels``, and any media fields already loaded on the sample.
        """
        return {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels, dtype=torch.long),
            **self._model_media,
        }

    def set_empty(self) -> None:
        """Mark this sample as empty so downstream packing can skip it.

        Returns:
            None.
        """
        empty = empty_model_sample()
        self._input_ids = empty["input_ids"].tolist()
        self._attention_mask = empty["attention_mask"].tolist()
        self._labels = empty["labels"].tolist()
        self._media_slots = []
        self._model_media = {}


class MLLMPack(list):
    """Packed training unit made from one or more tokenized MLLM samples.

    Attributes:
        samples: Source samples assigned to this pack.
        media_handler: Media handler shared by the source samples.
    """

    samples: list[MLLMSample]
    media_handler: MLLMMediaHandler

    def __init__(self, samples: Sequence[MLLMSample]) -> None:
        """Store source samples for late media resolution and final model-input merge.

        Args:
            samples: Tokenized source samples assigned to this pack.
        """
        self.samples = list(samples)
        self.media_handler = self.samples[0].media_handler
        super().__init__(self.samples)

    @property
    def source_sample_num(self) -> int:
        """Return how many source samples this pack represents.

        Returns:
            Number of source samples in the pack.
        """
        return len(self.samples)

    def to_model_inputs(self) -> dict[str, Any]:
        """Load media and merge source samples into one packed model-input dict.

        Returns:
            A packed model-input dictionary with concatenated token fields,
            ``pack_segment_ids``, ``source_sample_num``, and merged media fields.
        """
        samples = [sample.load_media() for sample in self.samples if sample.token_length > 0]
        samples = [sample for sample in samples if sample.token_length > 0]
        if not samples:
            empty = empty_model_sample()
            empty["pack_segment_ids"] = torch.empty(0, dtype=torch.long)
            empty["source_sample_num"] = 0
            return empty

        input_tensors = [torch.tensor(sample.input_ids, dtype=torch.long) for sample in samples]
        label_tensors = [torch.tensor(sample.labels, dtype=torch.long) for sample in samples]
        mask_tensors = [torch.tensor(sample.attention_mask, dtype=torch.long) for sample in samples]
        packed_sample = {
            "input_ids": torch.cat(input_tensors, dim=0),
            "attention_mask": torch.cat(mask_tensors, dim=0),
            "labels": torch.cat(label_tensors, dim=0),
            "pack_segment_ids": torch.cat(
                [
                    torch.full((sample.token_length,), fill_value=index + 1, dtype=torch.long)
                    for index, sample in enumerate(samples)
                ],
                dim=0,
            ),
            "source_sample_num": len(samples),
        }
        packed_sample.update(self.media_handler.merge_pack([sample.to_model_inputs() for sample in samples]))
        return packed_sample
