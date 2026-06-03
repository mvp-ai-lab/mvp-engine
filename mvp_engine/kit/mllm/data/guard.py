"""Data guards for filtering malformed multimodal samples."""

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
    """Filter malformed multimodal samples before expensive downstream processing."""

    def __init__(
        self,
        check_basic_formats: bool = True,
        check_input_ids: bool = False,
        check_image_sizes: bool = False,
        verbose: bool = True,
    ):
        """Configure which sample checks are active."""
        super().__init__()
        self.check_basic_formats = check_basic_formats
        self.check_input_ids = check_input_ids
        self.check_image_sizes = check_image_sizes
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

            images = sample.get("images")
            if isinstance(images, (list, tuple)):
                sample_info["image_count"] = len(images)
            elif images is not None:
                sample_info["image_count"] = 1

            image_size = sample.get("image_size")
            if image_size is None:
                image_size = sample.get("img_size")
            if isinstance(image_size, (list, tuple)):
                sample_info["image_size_count"] = len(image_size)
            elif image_size is not None:
                sample_info["image_size_count"] = 1

            input_ids = sample.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                sample_info["input_ids_shape"] = tuple(input_ids.shape)
        elif isinstance(sample, list):
            sample_info = {"list_sample_size": len(sample)}
        else:
            sample_info = repr(sample)
            if len(sample_info) > 200:
                sample_info = sample_info[:200] + "..."

        simple_info(
            f"Data guard skip: reason={result.reason} sample={sample_info}",
            level="warning",
        )

    def check(self, sample: Any) -> CheckResult:
        """Validate one sample against the enabled checks."""
        if not isinstance(sample, dict):
            return CheckResult(is_valid=False, reason="guard.not_dict")

        if self.check_basic_formats:
            messages = sample.get("messages") or sample.get("conversations")
            if not isinstance(messages, list):
                return CheckResult(is_valid=False, reason="guard.invalid_messages")
            if not isinstance(sample.get("images"), list):
                return CheckResult(is_valid=False, reason="guard.invalid_images")

        if self.check_input_ids and sample["input_ids"].size(0) <= 0:
            return CheckResult(is_valid=False, reason="guard.empty_input_ids")

        if self.check_image_sizes:
            images = sample.get("images", [])
            image_size = sample.get("img_size", []) or sample.get("image_size", [])
            if image_size is None:
                if len(images) == 0:
                    sample["image_size"] = []
                    return CheckResult(is_valid=True)
                return CheckResult(is_valid=False, reason="guard.missing_image_size")
            if not isinstance(image_size, (list, tuple)):
                return CheckResult(is_valid=False, reason="guard.invalid_image_size")
            if len(image_size) == 0:
                if len(images) == 0:
                    sample["image_size"] = []
                    return CheckResult(is_valid=True)
                return CheckResult(is_valid=False, reason="guard.missing_image_size")
            if not all(isinstance(size, (list, tuple)) for size in image_size):
                return CheckResult(is_valid=False, reason="guard.invalid_image_size")

            for size in image_size:
                if len(size) != 2 or not all(isinstance(dim, int) and dim > 0 for dim in size):
                    return CheckResult(is_valid=False, reason="guard.invalid_image_size")
            if len(image_size) != len(images):
                return CheckResult(is_valid=False, reason="guard.image_size_count_mismatch")

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
