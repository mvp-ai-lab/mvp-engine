"""JSONL dataset utilities for the minimal VLM recipe."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from mvp_dataset import Dataset
from mvp_dataset.core import RuntimeContext
from omegaconf import OmegaConf

from mvp_engine.distributed.utils import get_world_size

IMAGE_PLACEHOLDER = "<image>"


def _sample_location(sample: dict[str, Any]) -> tuple[Path, int]:
    """Extract the source JSONL path and 1-based row number from a sample.

    Args:
        sample: One row emitted by ``mvp_dataset.Dataset.from_jsonl(...)``.

    Returns:
        A tuple of ``(source_file, line_number)`` for error reporting.
    """
    source_file = sample.get("__file__")
    index_in_file = sample.get("__index_in_file__")
    if not isinstance(source_file, str) or not source_file:
        raise ValueError("mvp_dataset JSONL samples must include a string `__file__` field.")
    if not isinstance(index_in_file, int) or index_in_file < 0:
        raise ValueError("mvp_dataset JSONL samples must include a non-negative integer `__index_in_file__` field.")

    return Path(source_file).expanduser().resolve(), index_in_file + 1


def process_image(image: str, *, image_root: Path | None = None) -> Path:
    """Resolve one image reference from the JSONL row into an absolute path.

    Args:
        image: Image path stored in the JSONL row.
        image_root: Optional base directory used for relative image paths.

    Returns:
        The resolved absolute path to the image file.
    """
    if not image:
        raise ValueError(f"contains an invalid image path: {image!r}")

    resolved = Path(image).expanduser()
    if not resolved.is_absolute() and image_root is not None:
        resolved = image_root / resolved
    resolved = resolved.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"references missing image: {resolved}")
    return resolved


def process_message(
    message: dict[str, Any],
    image_iter: Iterable[Path],
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

    blocks: list[dict[str, str]] = []
    segments = content.split(image_placeholder)
    for i, segment in enumerate(segments):
        if segment:
            blocks.append({"type": "text", "text": segment})
        if i < len(segments) - 1:
            try:
                blocks.append({"type": "image", "image": str(next(image_iter))})
            except StopIteration as exc:
                raise ValueError("has more image placeholders than image paths.") from exc

    return {"role": role, "content": blocks}


def _tokenize_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    max_length: int,
) -> torch.Tensor:
    """Tokenize one conversation fragment with the recipe chat template.

    Args:
        processor: Hugging Face processor for Qwen3-VL.
        messages: Conversation fragment to tokenize.
        add_generation_prompt: Whether to append the assistant generation prompt.
        max_length: Maximum tokenized length after truncation.

    Returns:
        The 1D ``input_ids`` tensor for the provided conversation fragment.
    """
    tokenized = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return tokenized["input_ids"][0]


def build_labels(
    processor: Any,
    messages: list[dict[str, Any]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_length: int,
    ignore_index: int,
) -> torch.Tensor:
    """Build supervised labels for one tokenized conversation.

    Args:
        processor: Hugging Face processor for Qwen3-VL.
        messages: Rendered conversation in HF chat format.
        input_ids: Token ids for the full conversation.
        attention_mask: Attention mask for the full conversation.
        max_length: Maximum tokenized length after truncation.
        ignore_index: Label value used to mask tokens out of the loss.

    Returns:
        A label tensor where only assistant response tokens are supervised.
    """
    labels = torch.full_like(input_ids, ignore_index)
    valid_length = int(attention_mask.sum().item())
    if valid_length <= 0:
        return labels

    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        prefix_length = 0
        if message_index > 0:
            prefix_ids = _tokenize_messages(
                processor,
                messages[:message_index],
                add_generation_prompt=True,
                max_length=max_length,
            )
            prefix_length = int(prefix_ids.size(0))

        upto_ids = _tokenize_messages(
            processor,
            messages[: message_index + 1],
            add_generation_prompt=False,
            max_length=max_length,
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
) -> dict[str, Any]:
    """Validate one JSONL row and convert it into training tensors.

    Args:
        sample: One raw JSONL row emitted by ``mvp_dataset``.
        processor: Hugging Face processor for Qwen3-VL.
        max_length: Maximum tokenized sequence length.
        image_placeholder: Placeholder token that marks image positions in text.
        ignore_index: Label value used to mask tokens out of the loss.

    Returns:
        A processed sample containing token tensors and optional vision tensors.
    """
    if not isinstance(sample, dict):
        raise ValueError(f"Expected a dictionary sample, got {type(sample).__name__}.")

    source_file, line_number = _sample_location(sample)
    loc = f"{source_file}:{line_number}"

    try:
        messages = sample.get("messages")
        images = sample.get("images", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("has invalid `messages`.")
        if not isinstance(images, list):
            raise ValueError("has invalid `images`.")

        resolved_images = [process_image(p, image_root=source_file.parent) for p in images]

        image_iter = iter(resolved_images)
        rendered_messages = [process_message(msg, image_iter, image_placeholder=image_placeholder) for msg in messages]

        unused = list(image_iter)
        if unused:
            raise ValueError(f"has {len(unused)} unused image path(s).")

        model_inputs = processor.apply_chat_template(
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
        labels = build_labels(
            processor,
            rendered_messages,
            input_ids,
            attention_mask,
            max_length=max_length,
            ignore_index=ignore_index,
        )
    except Exception as exc:
        raise type(exc)(f"{loc} {exc}") from exc

    processed_sample: dict[str, Any] = {
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
    dataset_path_value = config.data.train_path
    if dataset_path_value is None:
        raise ValueError("Missing `data.train_path` for the minimal VLM recipe.")
    dataset_path = Path(dataset_path_value).expanduser().resolve()

    num_workers = int(config.data.num_workers)
    jsonl_num_shards = OmegaConf.select(config, "data.jsonl_num_shards")
    if jsonl_num_shards is None:
        jsonl_num_shards = max(get_world_size() * max(num_workers, 1), 1)

    output_dir = dataset_path.parent / ".jsonl_shards"
    context = RuntimeContext.from_runtime(seed=int(OmegaConf.select(config, "project.seed", default=42)))

    return (
        Dataset.from_jsonl(
            dataset_path,
            context=context,
            resample=True,
            num_shards=int(jsonl_num_shards),
            output_dir=output_dir,
        )
        .map(
            partial(
                process_sample,
                processor=processor,
                max_length=int(config.data.max_seq_len),
            )
        )
        .shuffle(buffer_size=int(OmegaConf.select(config, "data.shuffle_buffer", default=128)))
    )
