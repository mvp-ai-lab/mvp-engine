"""Stateful sample and packed-sample objects for text-only LM data pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from mvp_dataset.core import Assembler
from mvp_dataset.core.resume import stable_fingerprint

from mvp_engine.utils.log import simple_info

from .schema import LLMSchemaHandler
from .spec import LLMSampleSpec
from .tokenization import LLMTokenizationHandler


@dataclass(slots=True, init=False)
class LLMSample:
    """Own the lifecycle state for one tokenized text sample or document chunk."""

    _raw: dict[str, Any]
    schema_handler: LLMSchemaHandler
    tokenization_handler: LLMTokenizationHandler
    _metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _input_ids: list[int] | None = field(default=None, init=False, repr=False)
    _labels: list[int] | None = field(default=None, init=False, repr=False)
    _attention_mask: list[int] | None = field(default=None, init=False, repr=False)

    def __init__(
        self,
        raw: Mapping[str, Any],
        *,
        schema_handler: LLMSchemaHandler,
        tokenization_handler: LLMTokenizationHandler,
        metadata: Mapping[str, Any] | None = None,
        input_ids: Sequence[int] | None = None,
        labels: Sequence[int] | None = None,
        attention_mask: Sequence[int] | None = None,
    ) -> None:
        """Store raw row, handlers, and optional materialized token fields."""
        self._raw = dict(raw)
        self.schema_handler = schema_handler
        self.tokenization_handler = tokenization_handler
        self._metadata = dict(metadata or {})
        self._input_ids = list(input_ids) if input_ids is not None else None
        self._labels = list(labels) if labels is not None else None
        self._attention_mask = list(attention_mask) if attention_mask is not None else None

    def __getattr__(self, name: str) -> Any:
        """Read source-row fields as sample attributes."""
        try:
            raw = object.__getattribute__(self, "_raw")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        try:
            return raw[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        """Write source-row fields as sample attributes."""
        if name.startswith("_") or name in {"schema_handler", "tokenization_handler"}:
            object.__setattr__(self, name, value)
            return
        raw = object.__getattribute__(self, "_raw")
        raw[name] = value

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *, sample_spec: LLMSampleSpec) -> LLMSample:
        """Build a lazy sample from one raw source row and sample spec."""
        return cls(
            raw=row,
            schema_handler=sample_spec.schema_handler,
            tokenization_handler=sample_spec.tokenization_handler,
        )

    @classmethod
    def from_tokens(
        cls,
        row: Mapping[str, Any],
        *,
        sample_spec: LLMSampleSpec,
        metadata: Mapping[str, Any],
        input_ids: Sequence[int],
        labels: Sequence[int],
        attention_mask: Sequence[int],
    ) -> LLMSample:
        """Build a materialized sample from one tokenized document chunk."""
        return cls(
            raw=row,
            schema_handler=sample_spec.schema_handler,
            tokenization_handler=sample_spec.tokenization_handler,
            metadata=metadata,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    @property
    def input_ids(self) -> list[int]:
        """Token ids, tokenized lazily on first access."""
        self.tokenize()
        return self._input_ids or []

    @property
    def labels(self) -> list[int]:
        """Training labels aligned with ``input_ids``."""
        self.tokenize()
        return self._labels or []

    @property
    def attention_mask(self) -> list[int]:
        """1D attention mask aligned with ``input_ids``."""
        self.tokenize()
        return self._attention_mask or []

    @property
    def token_length(self) -> int:
        """Return the materialized token sequence length."""
        return len(self.input_ids)

    def tokenize(self) -> LLMSample:
        """Materialize the first token chunk when this sample is still lazy."""
        if self._input_ids is not None:
            return self

        chunks = self.to_chunks()
        if not chunks:
            self.set_empty()
            return self

        first = chunks[0]
        self._metadata = dict(first._metadata)
        self._input_ids = list(first.input_ids)
        self._labels = list(first.labels)
        self._attention_mask = list(first.attention_mask)
        return self

    def to_chunks(self) -> list[LLMSample]:
        """Normalize and tokenize this row into max-length sample chunks."""
        if self._input_ids is not None:
            return [self] if self._input_ids else []

        try:
            segments, metadata = self.schema_handler.normalize(self._raw)
            tokenized_chunks = self.tokenization_handler.tokenize(segments)
            chunks = [
                LLMSample(
                    raw=self._raw,
                    schema_handler=self.schema_handler,
                    tokenization_handler=self.tokenization_handler,
                    metadata={**metadata, "chunk_index": chunk_index},
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                for chunk_index, (input_ids, labels, attention_mask) in enumerate(tokenized_chunks)
            ]
        except Exception as exc:
            simple_info(exc, level="debug")
            return []
        return chunks

    def to_model_inputs(self) -> dict[str, Any]:
        """Return this sample as token tensors."""
        return {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels, dtype=torch.long),
        }

    def set_empty(self) -> None:
        """Mark this sample empty so downstream guards can drop it."""
        self._input_ids = []
        self._attention_mask = []
        self._labels = []


class LLMSampleAssembler(Assembler[dict[str, Any], LLMSample]):
    """Convert raw rows into one or more tokenized ``LLMSample`` chunks."""

    def __init__(self, sample_spec: LLMSampleSpec, assemble_context: Any | None = None) -> None:
        """Store the sample spec used to normalize and tokenize rows."""
        del assemble_context
        self.sample_spec = sample_spec

    def push(self, row: dict[str, Any]) -> list[LLMSample]:
        """Emit tokenized chunks for one raw row."""
        return LLMSample.from_row(row, sample_spec=self.sample_spec).to_chunks()

    def finish(self, *, drop_last: bool = False) -> list[LLMSample]:
        """No buffered state to flush."""
        del drop_last
        return []

    def state_dict(self) -> dict[str, object]:
        """Return resumable state."""
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore resumable state."""
        if state:
            raise ValueError(f"{self.__class__.__name__} does not have resumable state.")

    def fingerprint(self) -> str:
        """Return a stable sample-assembler fingerprint."""
        return stable_fingerprint(
            {
                "kind": self.__class__.__name__,
                "schema_handler": repr(self.sample_spec.schema_handler),
                "tokenization_handler": repr(self.sample_spec.tokenization_handler),
            }
        )


@dataclass(frozen=True, slots=True)
class LLMPack:
    """One fixed-length token-stream chunk emitted by sequential packing."""

    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    pack_segment_ids: list[int]
    source_sample_num: int
    position_ids: list[int] | None = None

    def to_model_inputs(self) -> dict[str, Any]:
        """Return this stream chunk as model-input tensors."""
        model_inputs: dict[str, Any] = {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels, dtype=torch.long),
            "pack_segment_ids": torch.tensor(self.pack_segment_ids, dtype=torch.long),
            "source_sample_num": int(self.source_sample_num),
        }
        if self.position_ids is not None:
            model_inputs["position_ids"] = torch.tensor(self.position_ids, dtype=torch.long)
        return model_inputs
