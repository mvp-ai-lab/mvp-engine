"""Validation guards for text-only LM data boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from mvp_dataset.core import Assembler
from mvp_dataset.core.resume import stable_fingerprint

from mvp_engine.utils.log import simple_info

from .sample import LLMSample
from .schema import LLMPretrainTextSchemaHandler, LLMSchemaHandler


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of one data-guard validation pass."""

    is_valid: bool
    reason: str | None = None


class LLMGuard(Assembler[Any, Any]):
    """Base mvp-dataset assembler that drops invalid items at one pipeline boundary."""

    def __init__(self, assemble_context: Any | None = None, *, verbose: bool = True) -> None:
        """Configure compact skip logging."""
        del assemble_context
        self.verbose = bool(verbose)

    def check(self, sample: Any) -> CheckResult:
        """Validate one item."""
        raise NotImplementedError

    def push(self, sample: Any) -> Iterable[Any]:
        """Emit the item only when it passes validation."""
        result = self.check(sample)
        if result.is_valid:
            return [sample]
        if self.verbose and result.reason:
            simple_info(f"Data guard skip: reason={result.reason} sample={self._sample_info(sample)}", level="warning")
        return []

    def finish(self, *, drop_last: bool = False) -> Iterable[Any]:
        """Flush guard state."""
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
        """Return a stable guard fingerprint."""
        return stable_fingerprint({"kind": self.__class__.__name__})

    @staticmethod
    def _sample_info(sample: Any) -> dict[str, Any] | str:
        """Return compact row metadata for skip logs."""
        if isinstance(sample, LLMSample):
            sample = sample._raw
        if isinstance(sample, dict):
            return {
                key: sample[key]
                for key in ("id", "source", "__source__", "__key__", "__global_index__")
                if key in sample
            }
        text = repr(sample)
        return text if len(text) <= 200 else text[:200] + "..."


class LLMRawRowGuard(LLMGuard):
    """Drop raw rows that cannot become text-LM samples."""

    def __init__(
        self,
        assemble_context: Any | None = None,
        *,
        schema_handler: LLMSchemaHandler | None = None,
        verbose: bool = True,
    ) -> None:
        """Store the schema handler used for cheap row validation."""
        super().__init__(assemble_context, verbose=verbose)
        self.schema_handler = schema_handler or LLMPretrainTextSchemaHandler()

    def check(self, sample: Any) -> CheckResult:
        """Validate the raw-row boundary before sample construction."""
        if not isinstance(sample, dict):
            return CheckResult(False, "raw.not_dict")
        reason = self.schema_handler.check_row(sample)
        if reason is not None:
            return CheckResult(False, reason)
        return CheckResult(True)

    def fingerprint(self) -> str:
        """Return a stable guard fingerprint."""
        return stable_fingerprint(
            {
                "kind": self.__class__.__name__,
                "schema_handler": repr(self.schema_handler),
            }
        )


class LLMSampleGuard(LLMGuard):
    """Drop tokenized samples that cannot enter packing."""

    def check(self, sample: Any) -> CheckResult:
        """Validate the tokenized sample boundary."""
        if not isinstance(sample, LLMSample):
            return CheckResult(False, "sample.invalid_type")
        sample.tokenize()
        if sample.token_length <= 0:
            return CheckResult(False, "sample.empty_tokens")
        if not any(label != sample.tokenization_handler.ignore_index for label in sample.labels):
            return CheckResult(False, "sample.no_supervised_tokens")
        return CheckResult(True)


class LLMModelInputGuard(LLMGuard):
    """Drop finalized packed model inputs that are empty or malformed."""

    def __init__(
        self,
        assemble_context: Any | None = None,
        *,
        ignore_index: int = -100,
        verbose: bool = True,
    ) -> None:
        """Store model-input validation options."""
        super().__init__(assemble_context, verbose=verbose)
        self.ignore_index = int(ignore_index)

    def check(self, sample: Any) -> CheckResult:
        """Validate finalized tensor fields."""
        if not isinstance(sample, dict):
            return CheckResult(False, "model_input.not_dict")

        for key in ("input_ids", "attention_mask", "labels", "pack_segment_ids"):
            value = sample.get(key)
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() <= 0:
                return CheckResult(False, f"model_input.invalid_{key}")

        if sample["input_ids"].shape != sample["attention_mask"].shape:
            return CheckResult(False, "model_input.attention_shape_mismatch")
        if sample["input_ids"].shape != sample["labels"].shape:
            return CheckResult(False, "model_input.label_shape_mismatch")
        if sample["input_ids"].shape != sample["pack_segment_ids"].shape:
            return CheckResult(False, "model_input.pack_segment_shape_mismatch")
        if int(sample.get("source_sample_num", 0)) <= 0:
            return CheckResult(False, "model_input.invalid_source_sample_num")
        if not sample["labels"].ne(self.ignore_index).any().item():
            return CheckResult(False, "model_input.no_supervised_tokens")
        return CheckResult(True)

    def fingerprint(self) -> str:
        """Return a stable guard fingerprint including label policy."""
        return stable_fingerprint({"kind": self.__class__.__name__, "ignore_index": self.ignore_index})
