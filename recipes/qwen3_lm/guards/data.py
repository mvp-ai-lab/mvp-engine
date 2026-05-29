"""Data guards for Qwen3 LM training."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from mvp_engine.utils.log import simple_info

try:
    from mvp_dataset.core import Assembler, RuntimeContext
except ImportError:  # pragma: no cover - only used before mvp_dataset is installed.

    class Assembler:
        """Fallback base class for JSONL-only local smoke tests."""

    class RuntimeContext:
        """Fallback runtime context for JSONL-only local smoke tests."""


def build_empty_sample():
    """Build an empty model-input sentinel for invalid samples."""
    return {
        "input_ids": torch.empty(0, dtype=torch.long),
        "attention_mask": torch.empty(0, dtype=torch.long),
        "labels": torch.empty(0, dtype=torch.long),
    }


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of one data-guard validation pass."""

    is_valid: bool
    reason: str | None = None


class DataGuard(Assembler):
    """Filter malformed Qwen3 LM samples before expensive downstream processing."""

    check_basic_formats: bool = True
    check_input_ids: bool = False
    record: bool = True

    def __init__(
        self,
        check_basic_formats: bool = True,
        check_input_ids: bool = False,
        verbose: bool = True,
    ):
        """Configure which Qwen3 LM sample checks are active."""
        super().__init__()
        self.check_basic_formats = check_basic_formats
        self.check_input_ids = check_input_ids
        self.verbose = verbose

    def _print_skip(self, result: CheckResult, sample: Any) -> None:
        """Log a compact description for one skipped sample."""
        if not self.verbose or result.reason is None:
            return

        if isinstance(sample, dict):
            sample_info: dict[str, Any] = {}
            for key in ("id", "source", "__source__", "__key__", "__global_index__"):
                if key in sample:
                    sample_info[key] = sample[key]

            messages = sample.get("messages") or sample.get("conversations")
            if isinstance(messages, list):
                sample_info["message_count"] = len(messages)
            input_ids = sample.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                sample_info["input_ids_shape"] = tuple(input_ids.shape)
        elif isinstance(sample, list):
            sample_info = {"list_sample_size": len(sample)}
        else:
            sample_info = repr(sample)
            if len(sample_info) > 200:
                sample_info = sample_info[:200] + "..."

        simple_info(f"Data guard skip: reason={result.reason} sample={sample_info}", level="warning")

    def check(self, sample: Any) -> CheckResult:
        """Validate one sample against the enabled checks."""
        if not isinstance(sample, dict):
            return CheckResult(is_valid=False, reason="guard.not_dict")

        if self.check_basic_formats:
            has_messages = isinstance(sample.get("messages") or sample.get("conversations"), list)
            has_prompt_response = isinstance(sample.get("prompt"), str) and isinstance(sample.get("response"), str)
            has_tokens = isinstance(sample.get("input_ids"), list)
            if not (has_messages or has_prompt_response or has_tokens):
                return CheckResult(is_valid=False, reason="guard.invalid_text_sample")

        if self.check_input_ids and sample["input_ids"].size(0) <= 0:
            return CheckResult(is_valid=False, reason="guard.empty_input_ids")

        return CheckResult(is_valid=True)

    def push(self, sample: Any) -> Iterable[Any]:
        """Validate a single sample and emit zero or one samples."""
        if isinstance(sample, list):
            filtered_sample: list[dict[str, Any]] = []
            for item in sample:
                result = self.check(item)
                if not result.is_valid:
                    self._print_skip(result, item)
                    continue
                filtered_sample.append(item)

            if not filtered_sample:
                self._print_skip(CheckResult(is_valid=False, reason="guard.empty_pack"), sample)
                return []
            return [filtered_sample]

        result = self.check(sample)
        if not result.is_valid:
            self._print_skip(result, sample)
            return []
        return [sample]

    def finish(self, *, drop_last: bool = False) -> Iterable[Any]:
        """Flush buffered samples at the end of assembly."""
        del drop_last
        return []


def build_dataguard(
    assemble_context: RuntimeContext,
    check_basic_formats: bool = True,
    check_input_ids: bool = False,
    record: bool = True,
):
    """Create a ``DataGuard`` assembler for the dataset assembly pipeline."""
    _ = assemble_context
    return DataGuard(
        check_basic_formats=check_basic_formats,
        check_input_ids=check_input_ids,
        verbose=record,
    )
