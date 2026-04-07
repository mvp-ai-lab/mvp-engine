"""Dataset processing utilities for the OpenBee recipe."""

from __future__ import annotations

import io
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from mvp_dataset import Dataset, set_logger
from mvp_dataset.core import RuntimeContext
from PIL import Image

from mvp_engine.utils.log import logger

from .packing import PackedSampleAssembler
from .types import ModelInputs

IMAGE_PLACEHOLDER = "<image>"
ROLE_MAP = {
    "assistant": "assistant",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "tool": "tool",
    "user": "user",
}


def process_image(
    image: str | bytes | dict[str, Any],
    *,
    image_root: Path | None = None,
) -> str | Image.Image:
    """Normalize one image input from one dataset row.

    Args:
        image: Image path, raw image bytes, or a parquet image struct with
            ``{"bytes", "path"}`` fields.
        image_root: Optional base directory used for relative image paths.

    Returns:
        Either an absolute image path string or a decoded RGB PIL image.
    """
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            return process_image(bytes(image_bytes), image_root=image_root)

        image_path = image.get("path")
        if isinstance(image_path, str) and image_path:
            return process_image(image_path, image_root=image_root)

        raise ValueError("contains an invalid image record.")

    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as decoded:
            return decoded.convert("RGB")

    if not isinstance(image, str):
        raise ValueError(f"contains an invalid image value: {type(image).__name__}.")
    if not image:
        raise ValueError(f"contains an invalid image path: {image!r}")

    resolved = Path(image).expanduser()
    if not resolved.is_absolute() and image_root is not None:
        resolved = image_root / resolved
    resolved = resolved.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"references missing image: {resolved}")
    return str(resolved)


def normalize_message(message: dict[str, Any]) -> dict[str, str]:
    """Normalize one source message from JSONL or parquet schema."""
    role = message.get("role")
    content = message.get("content")
    if isinstance(role, str) and isinstance(content, str) and role:
        return {"role": role, "content": content}

    source_role = message.get("from")
    source_content = message.get("value")
    normalized_role = ROLE_MAP.get(source_role)
    if normalized_role is None:
        raise ValueError(f"contains an invalid role: {source_role!r}")
    if not isinstance(source_content, str):
        raise ValueError("contains non-string content.")
    return {"role": normalized_role, "content": source_content}


def process_message(
    message: dict[str, Any],
    image_iter: Iterable[str | Image.Image],
    *,
    image_placeholder: str,
) -> dict[str, Any]:
    """Convert one source message into HF chat content blocks.

    Args:
        message: Source message in recipe JSONL format or parquet Open-Bee format.
        image_iter: Iterator over resolved image paths for ``<image>`` placeholders.
        image_placeholder: Placeholder token that marks image positions in text.

    Returns:
        A Hugging Face chat-format message with ``text`` and ``image`` blocks.
    """
    normalized_message = normalize_message(message)
    role = normalized_message["role"]
    content = normalized_message["content"]

    blocks: list[dict[str, Any]] = []
    segments = content.split(image_placeholder)
    for i, segment in enumerate(segments):
        if segment:
            blocks.append({"type": "text", "text": segment})
        if i < len(segments) - 1:
            try:
                image_value = next(image_iter)
                blocks.append({"type": "image", "image": image_value})
            except StopIteration as exc:
                raise ValueError("has more image placeholders than image paths.") from exc

    return {"role": role, "content": blocks}


def build_labels(
    apply_chat_template: Any,
    messages: list[dict[str, Any]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_length: int,
    ignore_index: int,
    skip_think_prefix: bool = False,
) -> torch.Tensor:
    """Build supervised labels for one tokenized conversation.

    Args:
        apply_chat_template: Bound ``processor.apply_chat_template`` callable.
        messages: Rendered conversation in HF chat format.
        input_ids: Token ids for the full conversation.
        attention_mask: Attention mask for the full conversation.
        max_length: Maximum tokenized length after truncation.
        ignore_index: Label value used to mask tokens out of the loss.
        skip_think_prefix: When True, the ``<think>…</think>`` opening block of
            each assistant turn is excluded from the loss, matching the behaviour
            of LLaMA-Factory's ``ReasoningTemplate`` with ``enable_thinking=False``.

    Returns:
        A label tensor where only assistant response tokens are supervised.
    """

    def tokenize_messages(
        conversation: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> torch.Tensor:
        """Tokenize one conversation fragment with the recipe chat template."""
        tokenized = apply_chat_template(
            [conversation],
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return tokenized["input_ids"][0]

    # Token IDs for the empty thinking block the model emits when not reasoning.
    # These match Qwen3-VL's tokenizer: <think> = 151667, \n\n = 271, </think> = 151668.
    # When skip_think_prefix=True we skip any leading run of these tokens so they
    # are not supervised — matching LLaMA-Factory's ReasoningTemplate behaviour.
    _THINK_PREFIX_IDS = {151667, 271, 151668}

    labels = torch.full_like(input_ids, ignore_index)
    valid_length = int(attention_mask.sum().item())
    if valid_length <= 0:
        return labels

    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        prefix_length = 0
        if message_index > 0:
            prefix_ids = tokenize_messages(
                messages[:message_index],
                add_generation_prompt=True,
            )
            prefix_length = int(prefix_ids.size(0))

        upto_ids = tokenize_messages(
            messages[: message_index + 1],
            add_generation_prompt=False,
        )
        upto_length = int(upto_ids.size(0))

        start = min(prefix_length, valid_length)
        end = min(upto_length, valid_length)

        # Optionally skip the leading <think>…</think> block from supervision.
        # Advance `start` past any consecutive think-prefix tokens so that only
        # the actual answer content contributes to the loss.
        if skip_think_prefix and start < end:
            pos = start
            while pos < end and input_ids[pos].item() in _THINK_PREFIX_IDS:
                pos += 1
            start = pos

        if start < end:
            labels[start:end] = input_ids[start:end]

    return labels


def process_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    max_length: int,
    image_placeholder: str = IMAGE_PLACEHOLDER,
    ignore_index: int = -100,
    skip_think_prefix: bool = False,
) -> ModelInputs:
    """Validate one dataset row and convert it into training tensors.

    Args:
        sample: One raw row emitted by ``mvp_dataset`` from JSONL or parquet.
        processor: Hugging Face processor (or compatible object with
            ``apply_chat_template`` and ``__fingerprint__``).
        max_length: Maximum tokenized sequence length.
        image_placeholder: Placeholder token that marks image positions in text.
        ignore_index: Label value used to mask tokens out of the loss.

    Returns:
        A processed sample containing token tensors and optional vision tensors.
    """
    if not isinstance(sample, dict):
        raise ValueError(f"Expected a dictionary sample, got {type(sample).__name__}.")

    apply_chat_template = processor.apply_chat_template

    source_file = sample.get("__file__")
    index_in_file = sample.get("__index_in_file__")
    if not isinstance(source_file, str) or not source_file:
        raise ValueError("mvp_dataset samples must include a string `__file__` field.")
    if not isinstance(index_in_file, int) or index_in_file < 0:
        raise ValueError("mvp_dataset samples must include a non-negative integer `__index_in_file__` field.")

    source_file = Path(source_file).expanduser().resolve()
    row_number = index_in_file + 1
    loc = f"{source_file}:{row_number}"

    try:
        # Validate the sample payload before we touch any file or tokenizer work.
        messages = sample.get("messages")
        if messages is None:
            messages = sample.get("conversations")
        images = sample.get("images", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("has invalid `messages`/`conversations`.")
        if not isinstance(images, list):
            raise ValueError("has invalid `images`.")

        # Normalize inline parquet image records or relative image paths.
        resolved_images = [process_image(p, image_root=source_file.parent) for p in images]

        # Rewrite each message into the processor's multimodal chat structure.
        image_iter = iter(resolved_images)
        rendered_messages = [process_message(msg, image_iter, image_placeholder=image_placeholder) for msg in messages]

        # Catch mismatches between declared images and <image> placeholders early.
        unused = list(image_iter)
        if unused:
            raise ValueError(f"has {len(unused)} unused image(s).")

        # Tokenize the full conversation once to get model inputs and vision features.
        model_inputs = apply_chat_template(
            [rendered_messages],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]

        # Re-tokenize assistant boundaries so only assistant responses contribute to loss.
        labels = build_labels(
            apply_chat_template,
            rendered_messages,
            input_ids,
            attention_mask,
            max_length=max_length,
            ignore_index=ignore_index,
            skip_think_prefix=skip_think_prefix,
        )
        if not torch.any(labels != ignore_index):
            raise ValueError("has no supervised assistant tokens after tokenization/truncation.")
    except Exception as exc:
        raise type(exc)(f"{loc} {exc}") from exc

    # Keep the processor outputs that the training step needs.
    processed_sample: ModelInputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if "pixel_values" in model_inputs:
        processed_sample["pixel_values"] = model_inputs["pixel_values"]
    if "image_grid_thw" in model_inputs:
        processed_sample["image_grid_thw"] = model_inputs["image_grid_thw"]
    return processed_sample


def build_dataset(config: Any, *, processor: Any):
    """Build the training dataset pipeline for the recipe.

    Args:
        config: Recipe config with dataset, runtime, and shuffle settings.
        processor: Hugging Face processor used during sample processing.

    Returns:
        An ``mvp_dataset.Dataset`` pipeline with parquet loading, processing, and
        sample-level shuffling.
    """
    set_logger(logger)
    dataset_path_value = config.data.train_path
    if dataset_path_value is None:
        raise ValueError("Missing `data.train_path` for the OpenBee recipe.")
    dataset_path = Path(dataset_path_value).expanduser().resolve()

    context = RuntimeContext.from_runtime(seed=int(config.seed))

    dataset = (
        Dataset.from_parquet(
            dataset_path,
            context=context,
            resample=True,
        )
        .map(
            partial(
                process_sample,
                processor=processor,
                max_length=int(config.data.max_seq_len),
                skip_think_prefix=not bool(config.data.enable_thinking),
            )
        )
        .shuffle(buffer_size=config.data.shuffle_buffer)
    )

    if config.data.packing:
        max_length = config.data.max_seq_len
        selection_strategy = config.data.packing_selection_strategy
        open_pack_limit = config.data.packing_open_pack_limit
        pack_buffer_size = config.data.packing_buffer_size

        dataset = dataset.assemble(
            lambda assemble_context: PackedSampleAssembler(
                max_length=max_length,
                selection_strategy=selection_strategy,
                open_pack_limit=open_pack_limit,
                pack_buffer_size=pack_buffer_size,
                seed=assemble_context.sample_shuffle_seed,
            )
        )

    if config.data.cache:
        dataset = dataset.cache(show_progress=config.data.cache_show_progress)

    return dataset
