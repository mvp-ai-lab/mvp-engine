"""Data guards for filtering malformed text-LM samples."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from mvp_dataset.core import Assembler

from mvp_engine.utils.log import simple_info


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of one data-guard validation pass."""

    is_valid: bool
    reason: str | None = None


class DataGuard(Assembler[Any, Any]):
    """Filter malformed text samples before expensive downstream processing."""

    def __init__(
        self,
        check_basic_formats: bool = True,
        check_input_ids: bool = False,
        text_field: str = "data",
        verbose: bool = True,
    ):
        """Configure which sample checks are active."""
        super().__init__()
        self.check_basic_formats = check_basic_formats
        self.check_input_ids = check_input_ids
        self.text_field = text_field
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
            input_ids = sample.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                sample_info["input_ids_shape"] = tuple(input_ids.shape)
        elif isinstance(sample, list):
            sample_info = {"list_sample_size": len(sample)}
        else:
            sample_info = repr(sample)[:200]

        simple_info(f"Data guard skip: reason={result.reason} sample={sample_info}", level="warning")

    def check(self, sample: Any) -> CheckResult:
        """Validate one sample against the enabled checks."""
        if not isinstance(sample, dict):
            return CheckResult(is_valid=False, reason="guard.not_dict")

        if self.check_basic_formats:
            text = sample.get(self.text_field)
            if not isinstance(text, str) or not text.strip():
                return CheckResult(is_valid=False, reason="guard.invalid_text")

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
