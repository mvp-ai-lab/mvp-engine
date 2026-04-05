"""JSONL dataset processing utilities for the OpenBee recipe."""

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

from mvp_engine.distributed.utils import get_world_size
from mvp_engine.utils.log import logger

from .packing import PackedSampleAssembler
from .types import ModelInputs

IMAGE_PLACEHOLDER = "<image>"


def process_image(
    image: str | bytes,
    *,
    image_root: Path | None = None,
) -> str | Image.Image:
    """Normalize one image input from the JSONL row.

    Args:
        image: Image path stored in the JSONL row, or raw bytes after
            ``Dataset.resolve_refs`` loads a tar-backed image reference.
        image_root: Optional base directory used for relative image paths.

    Returns:
        Either an absolute image path string or a decoded RGB PIL image.
    """
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


def process_message(
    message: dict[str, Any],
    image_iter: Iterable[str | Image.Image],
    *,
    image_placeholder: str,
) -> dict[str, Any]:
    """Convert one message into HF chat content blocks.

    Args:
        message: Source message with ``role`` and string ``content``.
        image_iter: Iterator over resolved image paths for ``<image>`` placeholders.
        image_placeholder: Placeholder token that marks image positions in text.

    Returns:
        A Hugging Face chat-format message with ``text`` and ``image`` blocks.
    """
    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not role:
        raise ValueError(f"contains an invalid role: {role!r}")
    if not isinstance(content, str):
        raise ValueError("contains non-string content.")

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
) -> torch.Tensor:
    """Build supervised labels for one tokenized conversation.

    Args:
        apply_chat_template: Bound ``processor.apply_chat_template`` callable.
        messages: Rendered conversation in HF chat format.
        input_ids: Token ids for the full conversation.
        attention_mask: Attention mask for the full conversation.
        max_length: Maximum tokenized length after truncation.
        ignore_index: Label value used to mask tokens out of the loss.

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
) -> ModelInputs:
    """Validate one JSONL row and convert it into training tensors.

    Args:
        sample: One raw JSONL row emitted by ``mvp_dataset``.
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
        raise ValueError("mvp_dataset JSONL samples must include a string `__file__` field.")
    if not isinstance(index_in_file, int) or index_in_file < 0:
        raise ValueError("mvp_dataset JSONL samples must include a non-negative integer `__index_in_file__` field.")

    source_file = Path(source_file).expanduser().resolve()
    line_number = index_in_file + 1
    loc = f"{source_file}:{line_number}"

    try:
        # Validate the sample payload before we touch any file or tokenizer work.
        messages = sample.get("messages")
        images = sample.get("images", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("has invalid `messages`.")
        if not isinstance(images, list):
            raise ValueError("has invalid `images`.")

        # Normalize image paths against the JSONL location so relative paths work.
        resolved_images = [process_image(p, image_root=source_file.parent) for p in images]

        # Rewrite each message into the processor's multimodal chat structure.
        image_iter = iter(resolved_images)
        rendered_messages = [process_message(msg, image_iter, image_placeholder=image_placeholder) for msg in messages]

        # Catch mismatches between declared images and <image> placeholders early.
        unused = list(image_iter)
        if unused:
            raise ValueError(f"has {len(unused)} unused image path(s).")

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
        An ``mvp_dataset.Dataset`` pipeline with JSONL loading, processing, and
        sample-level shuffling.
    """
    set_logger(logger)
    dataset_path_value = config.data.train_path
    if dataset_path_value is None:
        raise ValueError("Missing `data.train_path` for the OpenBee recipe.")
    dataset_path = Path(dataset_path_value).expanduser().resolve()

    output_dir = dataset_path.parent / ".jsonl_shards"
    context = RuntimeContext.from_runtime(seed=int(config.seed))
    num_shards = max(get_world_size(), 1)

    dataset = (
        Dataset.from_jsonl(
            dataset_path,
            context=context,
            resample=True,
            group_key="images",
            num_shards=int(num_shards),
            output_dir=output_dir,
        )
        .resolve_refs([("images", dataset_path.parent)])
        .map(
            partial(
                process_sample,
                processor=processor,
                max_length=int(config.data.max_seq_len),
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
