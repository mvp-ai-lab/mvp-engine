"""Validation and batch guards for MLLM data boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from mvp_dataset.core import Assembler

from mvp_engine.utils.log import simple_info

from .sample import MLLMSample


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of one data-guard validation pass.

    Attributes:
        is_valid: Whether the checked item should pass through.
        reason: Optional compact reason code used in skip logs.
    """

    is_valid: bool
    reason: str | None = None


class MLLMTextOnlyBatchGuard:
    """Add dummy media inputs when a collated MLLM batch has no media tensors.

    Attributes:
        dummy_inputs: Minimal valid model inputs used as the dummy media suffix.
        media_keys: Media tensor keys that should be present and non-empty.
        pad_token_id: Padding value for token ids.
        ignore_index: Label value used for dummy suffix labels.
    """

    def __init__(
        self,
        *,
        dummy_inputs: Mapping[str, torch.Tensor],
        media_keys: Sequence[str],
        pad_token_id: int,
        ignore_index: int = -100,
    ) -> None:
        """Store dummy token and media fields used for text-only batches.

        Args:
            dummy_inputs: Minimal valid model inputs containing token fields and the required media fields.
            media_keys: Media field names that must be non-empty for the model backend.
            pad_token_id: Token id used to pad dummy suffixes for other batch rows.
            ignore_index: Label value used for dummy suffix labels.

        Raises:
            ValueError: If ``dummy_inputs`` does not contain every required key.
        """
        self.dummy_inputs = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in dummy_inputs.items()
        }
        self.media_keys = tuple(media_keys)
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)

        required_keys = {"input_ids", "attention_mask", *self.media_keys}
        missing_keys = required_keys.difference(self.dummy_inputs)
        if missing_keys:
            raise ValueError(f"Dummy media inputs are missing keys: {sorted(missing_keys)}")

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Return a batch with dummy media tensors when it is text-only.

        Args:
            batch: Collated model-input batch.

        Returns:
            The original batch if media is present, otherwise a shallow-copied batch
            with one dummy media suffix inserted.
        """
        if any(isinstance(batch.get(key), torch.Tensor) and batch[key].numel() > 0 for key in self.media_keys):
            return batch

        dummy_input_ids = self.dummy_inputs["input_ids"].to(
            device=batch["input_ids"].device,
            dtype=batch["input_ids"].dtype,
        )
        dummy_attention_mask = self.dummy_inputs["attention_mask"].to(
            device=batch["attention_mask"].device,
            dtype=batch["attention_mask"].dtype,
        )
        batch_size = int(batch["input_ids"].shape[0])
        dummy_length = int(dummy_input_ids.numel())
        next_segment_id = int(batch["pack_segment_ids"][0].max().item()) + 1

        input_suffix = torch.full(
            (batch_size, dummy_length),
            fill_value=self.pad_token_id,
            dtype=batch["input_ids"].dtype,
            device=batch["input_ids"].device,
        )
        attention_suffix = torch.zeros(
            (batch_size, dummy_length),
            dtype=batch["attention_mask"].dtype,
            device=batch["attention_mask"].device,
        )
        label_suffix = torch.full(
            (batch_size, dummy_length),
            fill_value=self.ignore_index,
            dtype=batch["labels"].dtype,
            device=batch["labels"].device,
        )
        segment_suffix = torch.zeros(
            (batch_size, dummy_length),
            dtype=batch["pack_segment_ids"].dtype,
            device=batch["pack_segment_ids"].device,
        )
        input_suffix[0] = dummy_input_ids
        attention_suffix[0] = dummy_attention_mask
        segment_suffix[0] = next_segment_id

        batch = dict(batch)
        batch["input_ids"] = torch.cat([batch["input_ids"], input_suffix], dim=1)
        batch["attention_mask"] = torch.cat([batch["attention_mask"], attention_suffix], dim=1)
        batch["labels"] = torch.cat([batch["labels"], label_suffix], dim=1)
        batch["pack_segment_ids"] = torch.cat([batch["pack_segment_ids"], segment_suffix], dim=1)
        for key in self.media_keys:
            value = self.dummy_inputs[key]
            batch[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value

        batch["num_input_tokens"] = batch["attention_mask"].sum(dim=-1)
        shifted_labels = torch.nn.functional.pad(batch["labels"], (0, 1), value=self.ignore_index)[..., 1:]
        batch["num_loss_tokens"] = shifted_labels.ne(self.ignore_index).sum(dim=-1)
        batch["num_source_samples"] = batch["source_sample_num"].clone()
        batch["total_tokens"] = int(batch["num_input_tokens"].sum().item())
        batch["effective_tokens"] = int(batch["num_loss_tokens"].sum().item())
        return batch

    def fingerprint(self) -> str:
        """Return a stable fingerprint for loader-side map resume checks.

        Returns:
            Stable string fingerprint of dummy-media guard configuration.
        """
        media_shapes = {
            key: (tuple(value.shape), str(value.dtype))
            for key, value in self.dummy_inputs.items()
            if isinstance(value, torch.Tensor)
        }
        return repr(
            {
                "kind": self.__class__.__name__,
                "media_keys": self.media_keys,
                "pad_token_id": self.pad_token_id,
                "ignore_index": self.ignore_index,
                "media_shapes": media_shapes,
            }
        )


class MLLMGuard(Assembler[Any, Any]):
    """Base mvp-dataset assembler that drops invalid items at one pipeline boundary.

    Attributes:
        verbose: Whether invalid items are logged with compact skip metadata.
    """

    def __init__(self, assemble_context: Any | None = None, *, verbose: bool = True) -> None:
        """Configure compact skip logging.

        Args:
            assemble_context: Unused mvp-dataset assembler context.
            verbose: Whether invalid items should be logged.
        """
        del assemble_context
        self.verbose = bool(verbose)

    def check(self, sample: Any) -> CheckResult:
        """Validate one item.

        Args:
            sample: Item at this pipeline boundary.

        Returns:
            Validation result.
        """
        raise NotImplementedError

    def push(self, sample: Any) -> Iterable[Any]:
        """Emit the item only when it passes validation.

        Args:
            sample: Item at this pipeline boundary.

        Returns:
            A one-item iterable for valid input, or an empty iterable for invalid input.
        """
        result = self.check(sample)
        if result.is_valid:
            return [sample]
        if self.verbose and result.reason:
            simple_info(f"Data guard skip: reason={result.reason} sample={self._sample_info(sample)}", level="warning")
        return []

    def finish(self, *, drop_last: bool = False) -> Iterable[Any]:
        """Flush guard state.

        Args:
            drop_last: Unused assembler option accepted for interface compatibility.

        Returns:
            An empty iterable because guards do not buffer state.
        """
        del drop_last
        return []

    def state_dict(self) -> dict[str, object]:
        """Return resumable state.

        Returns:
            An empty state dictionary because guards are stateless.
        """
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore resumable state.

        Args:
            state: State dictionary to restore.

        Raises:
            ValueError: If non-empty state is provided.
        """
        if state:
            raise ValueError(f"{self.__class__.__name__} does not have resumable state.")

    def fingerprint(self) -> str:
        """Return a stable guard fingerprint.

        Returns:
            Stable string fingerprint for mvp-dataset resume checks.
        """
        return repr({"kind": self.__class__.__name__})

    @staticmethod
    def _sample_info(sample: Any) -> dict[str, Any] | str:
        """Return compact row metadata for skip logs."""
        if isinstance(sample, MLLMSample):
            sample = sample._raw
        if isinstance(sample, dict):
            info = {
                key: sample[key]
                for key in ("id", "source", "__source__", "__key__", "__global_index__")
                if key in sample
            }
            messages = sample.get("messages") or sample.get("conversations")
            if isinstance(messages, list):
                info["message_count"] = len(messages)
            images = sample.get("images")
            if isinstance(images, (list, tuple)):
                info["image_count"] = len(images)
            media = sample.get("media")
            if isinstance(media, (list, tuple)):
                info["media_count"] = len(media)
            return info
        text = repr(sample)
        return text if len(text) <= 200 else text[:200] + "..."


class MLLMRawRowGuard(MLLMGuard):
    """Drop raw rows that cannot become conversation MLLM samples.

    Attributes:
        verbose: Whether invalid rows are logged.
    """

    def check(self, sample: Any) -> CheckResult:
        """Validate the cheap raw-row boundary before sample construction.

        Args:
            sample: Raw source item.

        Returns:
            Validation result with a compact reason code on failure.
        """
        if not isinstance(sample, dict):
            return CheckResult(False, "raw.not_dict")

        messages = sample.get("messages") or sample.get("conversations")
        if not isinstance(messages, list) or not messages:
            return CheckResult(False, "raw.invalid_messages")
        for message in messages:
            if not isinstance(message, dict):
                return CheckResult(False, "raw.invalid_message")
            content = message.get("content") if "content" in message else message.get("value")
            if not isinstance(content, str):
                return CheckResult(False, "raw.invalid_message_content")

        media = sample.get("media")
        if media is not None:
            return self._check_media_entries(media)
        return self._check_images(sample)

    def _check_media_entries(self, media: Any) -> CheckResult:
        """Validate explicit media entries before schema normalization."""
        if not isinstance(media, (list, tuple)):
            return CheckResult(False, "raw.invalid_media")
        for entry in media:
            if not isinstance(entry, dict):
                return CheckResult(False, "raw.invalid_media_entry")
            media_type = entry.get("type", "image")
            if media_type != "image":
                return CheckResult(False, "raw.unsupported_media_type")
            if entry.get("value", entry.get(media_type)) is None:
                return CheckResult(False, "raw.missing_media_value")
            size = entry.get("size") or entry.get("image_size") or entry.get("img_size")
            if not self._valid_image_size(size):
                return CheckResult(False, "raw.invalid_image_size")
        return CheckResult(True)

    def _check_images(self, sample: dict[str, Any]) -> CheckResult:
        """Validate image-list rows and their image-size metadata."""
        images = sample.get("images", [])
        if images is None:
            images = []
        if not isinstance(images, (list, tuple)):
            return CheckResult(False, "raw.invalid_images")

        image_sizes = sample.get("img_size", []) or sample.get("image_size", [])
        if image_sizes is None:
            image_sizes = []
        if not isinstance(image_sizes, (list, tuple)):
            return CheckResult(False, "raw.invalid_image_size")
        if len(images) != len(image_sizes):
            return CheckResult(False, "raw.image_size_count_mismatch")
        if any(not self._valid_image_size(size) for size in image_sizes):
            return CheckResult(False, "raw.invalid_image_size")
        return CheckResult(True)

    @staticmethod
    def _valid_image_size(size: Any) -> bool:
        """Return whether one image-size record has positive integer width and height."""
        if isinstance(size, dict):
            width = size.get("width")
            height = size.get("height")
        elif isinstance(size, (list, tuple)) and len(size) >= 2:
            width = size[0]
            height = size[1]
        else:
            return False
        return isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0


class MLLMSampleGuard(MLLMGuard):
    """Tokenize samples and drop rows that cannot enter packing.

    Attributes:
        verbose: Whether invalid samples are logged.
    """

    def check(self, sample: Any) -> CheckResult:
        """Validate the tokenized sample boundary.

        Args:
            sample: Sample object after raw-row conversion.

        Returns:
            Validation result with a compact reason code on failure.
        """
        if not isinstance(sample, MLLMSample):
            return CheckResult(False, "sample.invalid_type")
        sample.tokenize()
        if sample.token_length <= 0:
            return CheckResult(False, "sample.empty_tokens")
        if not any(label != sample.tokenization_handler.ignore_index for label in sample.labels):
            return CheckResult(False, "sample.no_supervised_tokens")
        return CheckResult(True)


class MLLMModelInputGuard(MLLMGuard):
    """Drop finalized packed model inputs that are empty or malformed.

    Attributes:
        ignore_index: Label value that marks ignored tokens.
        verbose: Whether invalid model-input items are logged.
    """

    def __init__(
        self,
        assemble_context: Any | None = None,
        *,
        ignore_index: int = -100,
        verbose: bool = True,
    ) -> None:
        """Store model-input validation options.

        Args:
            assemble_context: Optional mvp-dataset assembler context.
            ignore_index: Label value that marks ignored tokens.
            verbose: Whether invalid model-input samples should be logged.
        """
        super().__init__(assemble_context, verbose=verbose)
        self.ignore_index = int(ignore_index)

    def check(self, sample: Any) -> CheckResult:
        """Validate finalized tensor fields.

        Args:
            sample: Packed model-input item.

        Returns:
            Validation result with a compact reason code on failure.
        """
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
        """Return a stable guard fingerprint.

        Returns:
            Stable string fingerprint including ``ignore_index``.
        """
        return repr({"kind": self.__class__.__name__, "ignore_index": self.ignore_index})
