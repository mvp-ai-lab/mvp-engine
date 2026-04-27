from collections.abc import Iterable
from typing import Any

import torch
from mvp_dataset.core import Assembler, RuntimeContext

from ..utils.data_logging import record_skip


def build_empty_sample():
    """Build an empty model-input sentinel for invalid samples.

    The OpenBee data pipeline represents dropped examples as zero-length
    tensors when a later stage needs a concrete ``ModelInputs`` object instead
    of simply filtering the sample out. Downstream code can identify this
    sentinel by checking that ``input_ids`` has length 0.

    Returns:
        A minimal ``ModelInputs`` mapping with empty ``input_ids``,
        ``attention_mask``, and ``labels`` tensors.
    """
    return {
        "input_ids": torch.empty(0, dtype=torch.long),
        "attention_mask": torch.empty(0, dtype=torch.long),
        "labels": torch.empty(0, dtype=torch.long),
    }


class DataGuard(Assembler[dict[str, Any], dict[str, Any]]):
    """Filter malformed OpenBee samples before expensive downstream processing.

    ``DataGuard`` is a lightweight dataset assembler. It receives one decoded
    sample at a time and either forwards it unchanged or returns an empty list
    to drop it from the stream. The enabled checks are intentionally controlled
    by flags so callers can use only the guards that make sense for the current
    pipeline stage.

    Attributes:
        check_basic_formats: Drop samples with basic format errors, such as missing or malformed
                             ``messages``/``conversations`` or ``images`` fields.
        check_input_ids: Drop samples whose ``input_ids`` tensor is empty.
        check_image_sizes: Drop samples with missing, malformed, or mismatched
                           ``image_size`` metadata.
        record: Whether guard drops should be counted as skips.
    """

    check_basic_formats: bool = True
    check_input_ids: bool = False
    check_image_sizes: bool = False
    record: bool = True

    def __init__(
        self,
        check_basic_formats: bool = True,
        check_input_ids: bool = False,
        check_image_sizes: bool = False,
        record: bool = True,
    ):
        super().__init__()
        self.check_basic_formats = check_basic_formats
        self.check_input_ids = check_input_ids
        self.check_image_sizes = check_image_sizes
        self.record = record

    def _record_skip(self, reason: str, sample: Any, detail: str | None = None) -> None:
        if self.record:
            record_skip(reason, sample, detail=detail)

    def push(self, sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Validate a single sample and emit zero or one samples.

        Args:
            sample: A decoded OpenBee sample. Expected fields depend on which
                checks are enabled.

        Returns:
            ``[sample]`` when the sample passes all enabled checks, or ``[]``
            when the sample should be skipped. Invalid samples are logged before
            they are dropped.
        """
        # =============================
        # Check basic formats
        if self.check_basic_formats:
            if not isinstance(sample, dict):
                self._record_skip("guard.not_dict", sample)
                return []
            messages = sample.get("messages") or sample.get("conversations")
            if not isinstance(messages, list):
                self._record_skip("guard.invalid_messages", sample)
                return []
            if not isinstance(sample.get("images"), list):
                self._record_skip("guard.invalid_images", sample)
                return []

        # =============================
        # Check if the sample has empty input_ids, which indicates an invalid sample.
        if self.check_input_ids and sample["input_ids"].size(0) <= 0:
            self._record_skip("guard.empty_input_ids", sample)
            return []

        # =============================
        # Check if the sample contains a "image_size" field and if it's valid.
        if self.check_image_sizes:
            images = sample.get("images", [])
            image_size = sample.get("img_size", []) or sample.get("image_size", [])
            if image_size is None:
                if len(images) == 0:
                    # If there are no images, it's fine to have no image_size.
                    sample["image_size"] = []
                    return [sample]

                self._record_skip("guard.missing_image_size", sample)
                return []
            if not isinstance(image_size, (list, tuple)):
                self._record_skip("guard.invalid_image_size", sample)
                return []
            if len(image_size) == 0:
                if len(images) == 0:
                    sample["image_size"] = []
                    return [sample]
                self._record_skip("guard.missing_image_size", sample)
                return []
            if not all(isinstance(size, (list, tuple)) for size in image_size):
                self._record_skip("guard.invalid_image_size", sample)
                return []

            for size in image_size:
                if len(size) != 2 or not all(isinstance(dim, int) and dim > 0 for dim in size):
                    self._record_skip("guard.invalid_image_size", sample, detail=f"image_size={size!r}")
                    return []
            if len(image_size) != len(images):
                self._record_skip(
                    "guard.image_size_count_mismatch",
                    sample,
                    detail=f"{len(image_size)} sizes vs {len(images)} images",
                )
                return []

        return [sample]

    def finish(self, *, drop_last: bool = False) -> Iterable[dict[str, Any]]:
        """Flush buffered samples at the end of assembly.

        ``DataGuard`` is stateless and never buffers samples, so there is
        nothing to emit during finalization.

        Args:
            drop_last: Accepted for the assembler interface; ignored because
                no partial batch or buffered state exists.

        Returns:
            An empty iterable.
        """
        del drop_last
        return []


def build_dataguard(
    assemble_context: RuntimeContext,
    check_basic_formats: bool = True,
    check_input_ids: bool = False,
    check_image_sizes: bool = False,
    record: bool = True,
):
    """Create a ``DataGuard`` assembler for the dataset assembly pipeline.

    Args:
        assemble_context: Runtime assembly context supplied by ``mvp_dataset``.
            It is currently unused because this guard has no worker-local setup.
        check_input_ids: Whether to drop samples with empty ``input_ids``.
        check_image_sizes: Whether to validate ``image_size`` metadata against
            the sample's image list.
        record: Whether guard drops should be counted as skips.

    Returns:
        A configured ``DataGuard`` instance.
    """
    _ = assemble_context
    return DataGuard(
        check_basic_formats=check_basic_formats,
        check_input_ids=check_input_ids,
        check_image_sizes=check_image_sizes,
        record=record,
    )
