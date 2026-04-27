from collections.abc import Iterable
from typing import Any

import torch
from mvp_dataset.core import Assembler, RuntimeContext

from mvp_engine.utils.log import simple_info


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
    """

    check_basic_formats: bool = True
    check_input_ids: bool = False
    check_image_sizes: bool = False

    def __init__(
        self,
        check_basic_formats: bool = True,
        check_input_ids: bool = False,
        check_image_sizes: bool = False,
    ):
        super().__init__()
        self.check_basic_formats = check_basic_formats
        self.check_input_ids = check_input_ids
        self.check_image_sizes = check_image_sizes

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
                simple_info(f"DataGuard: sample is not a dict, skipping sample: {sample}", level="warning")
                return []
            messages = sample.get("messages") or sample.get("conversations")
            if not isinstance(messages, list):
                simple_info(
                    f"DataGuard: missing or invalid messages/conversations field, skipping sample: {sample}",
                    level="warning",
                )
                return []
            if not isinstance(sample.get("images"), list):
                simple_info(f"DataGuard: missing or invalid images field, skipping sample: {sample}", level="warning")
                return []

        # =============================
        # Check if the sample has empty input_ids, which indicates an invalid sample.
        if self.check_input_ids and sample["input_ids"].size(0) <= 0:
            simple_info(f"DataGuard: empty input_ids, skipping sample: {sample}", level="warning")
            return []

        # =============================
        # Check if the sample contains a "image_size" field and if it's valid.
        if self.check_image_sizes:
            images = sample.get("images", [])
            image_size = sample.get("image_size")
            if image_size is None and "img_size" in sample:
                image_size = sample.get("img_size")  # Support both "image_size" and "img_size" keys.
            if image_size is None:
                if len(images) == 0:
                    # If there are no images, it's fine to have no image_size.
                    sample["image_size"] = []
                    return [sample]

                simple_info(f"DataGuard: missing image_size field, skipping sample: {sample}", level="warning")
                return []
            if not isinstance(image_size, (list, tuple)):
                simple_info(f"DataGuard: invalid image_size type, skipping sample: {sample}", level="warning")
                return []
            if len(image_size) == 0:
                if len(images) == 0:
                    sample["image_size"] = []
                    return [sample]
                simple_info(
                    f"DataGuard: empty image_size for non-text-only sample, skipping sample: {sample}",
                    level="warning",
                )
                return []
            if isinstance(image_size[0], (list, tuple)):
                # If image_size is a list of sizes, check each one.
                for size in image_size:
                    if (
                        not isinstance(size, (list, tuple))
                        or len(size) != 2
                        or not all(isinstance(dim, int) and dim > 0 for dim in size)
                    ):
                        simple_info(f"DataGuard: invalid image_size {size}, skipping sample: {sample}", level="warning")
                        return []
                if len(image_size) != len(images):
                    simple_info(
                        "DataGuard: image_size list length does not match images list length, "
                        f"skipping sample: {sample}",
                        level="warning",
                    )
                    return []
            else:
                # If image_size is a single size, check it directly.
                if len(image_size) != 2 or not all(isinstance(dim, int) and dim > 0 for dim in image_size):
                    simple_info(
                        f"DataGuard: invalid image_size {image_size}, skipping sample: {sample}",
                        level="warning",
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
):
    """Create a ``DataGuard`` assembler for the dataset assembly pipeline.

    Args:
        assemble_context: Runtime assembly context supplied by ``mvp_dataset``.
            It is currently unused because this guard has no worker-local setup.
        check_input_ids: Whether to drop samples with empty ``input_ids``.
        check_image_sizes: Whether to validate ``image_size`` metadata against
            the sample's image list.

    Returns:
        A configured ``DataGuard`` instance.
    """
    _ = assemble_context
    return DataGuard(
        check_basic_formats=check_basic_formats,
        check_input_ids=check_input_ids,
        check_image_sizes=check_image_sizes,
    )
