"""Dataset processing utilities for the PanguVL recipe."""

from __future__ import annotations

import glob
import io
import os
import re
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any

import imagesize
import torch
from mvp_dataset import Dataset, set_logger
from mvp_dataset.core import RuntimeContext
from mvp_dataset.utils.url import normalize_paths
from PIL import Image

from mvp_engine.utils.log import logger

from .gate import build_invalid_sample_gate_assembler, build_skipped_sample
from .packing import build_packed_sample_assembler
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

THOUGHT_PREFIX = "<think>\n"
THOUGHT_SUFFIX = "\n</think>\n\n"
THOUGHT_PATTERN = re.compile(f"{re.escape(THOUGHT_PREFIX)}(.*?){re.escape(THOUGHT_SUFFIX)}", re.DOTALL)
THOUGHT_MARKERS = (THOUGHT_PREFIX.strip(), THOUGHT_SUFFIX.strip())
MULTIMODAL_PLACEHOLDER = "<|panguvl_multimodal_placeholder|>"


def resolve_cache_dir(train_path: str, configured_cache_dir: str | None) -> Path:
    """Resolve the cache root for the current dataset specification.

    When ``configured_cache_dir`` is omitted, the cache is colocated with the
    resolved dataset root so recipe runs keep preprocessing artifacts near the
    source parquet files instead of under the launcher working directory.
    """
    if configured_cache_dir is not None:
        return Path(configured_cache_dir).expanduser().resolve()

    shard_paths = resolve_dataset_shards(train_path)
    shard_dirs = [str(Path(shard_path).expanduser().resolve().parent) for shard_path in shard_paths]
    dataset_root = Path(os.path.commonpath(shard_dirs))
    return (dataset_root / ".cache").resolve()


def resolve_dataset_shards(train_path: str) -> list[str]:
    """Resolve one dataset shard spec into concrete absolute parquet paths.

    ``mvp_dataset.normalize_paths`` expands ``*`` and brace ranges, but its glob
    call does not enable recursive ``**`` matching. PanguVL stage configs use
    nested parquet trees, so expand recursive globs here before constructing the
    dataset source list.
    """
    shard_specs = normalize_paths(train_path)
    shard_paths: list[str] = []

    for shard_spec in shard_specs:
        if any(char in shard_spec for char in "*?["):
            matches = sorted(glob.glob(shard_spec, recursive=True))
            if matches:
                shard_paths.extend(str(Path(match).expanduser().resolve()) for match in matches)
                continue
        shard_paths.append(str(Path(shard_spec).expanduser().resolve()))

    return shard_paths


def configure_cache_write_batch_size(batch_size: int) -> None:
    """Override ``mvp_dataset``'s cache write chunk size for large multimodal rows.

    ``mvp_dataset`` currently defaults to buffering 8192 samples per Lance write.
    PanguVL cache rows can contain long token tensors plus image features, so that
    default spikes host memory and can trigger OOM during cache creation.
    """
    import mvp_dataset.cache.store as cache_store

    original = getattr(cache_store, "_panguvl_original_write_lance_dataset", cache_store._write_lance_dataset)
    if not hasattr(cache_store, "_panguvl_original_write_lance_dataset"):
        cache_store._panguvl_original_write_lance_dataset = original

    def _write_lance_dataset_with_panguvl_batch_size(
        stream,
        uri,
        *,
        batch_size: int = batch_size,
        max_rows_per_group=None,
    ):
        return original(
            stream,
            uri,
            batch_size=batch_size,
            max_rows_per_group=max_rows_per_group,
        )

    cache_store._write_lance_dataset = _write_lance_dataset_with_panguvl_batch_size


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


def add_thought(content: str = "") -> str:
    """Match LF ReasoningTemplate.add_thought for Qwen3-VL thought words."""
    return f"{THOUGHT_PREFIX}{THOUGHT_SUFFIX}{content}"


def remove_thought(content: str) -> str:
    """Match LF ReasoningTemplate.remove_thought semantics."""
    return THOUGHT_PATTERN.sub("", content).lstrip("\n")


def extract_thought(content: str) -> str | None:
    """Match LF ReasoningTemplate.extract_thought semantics."""
    match = THOUGHT_PATTERN.search(content)
    if match is None:
        return None
    return match.group(1)


def is_thought_empty(content: str) -> bool:
    """Match LF ReasoningTemplate.is_thought_empty semantics."""
    thought = extract_thought(content)
    if thought is None:
        return True
    return not thought.strip()


def _flatten_message_content(content: Any) -> tuple[str, list[dict[str, Any]], bool]:
    """Flatten multimodal message content into text plus ordered non-text blocks."""
    if isinstance(content, str):
        return content, [], False
    if not isinstance(content, list):
        return "", [], False

    flat_parts: list[str] = []
    non_text_blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            flat_parts.append(block["text"])
        elif isinstance(block, dict):
            flat_parts.append(MULTIMODAL_PLACEHOLDER)
            non_text_blocks.append(block)
    return "".join(flat_parts), non_text_blocks, True


def _rebuild_message_content(
    flat_text: str,
    non_text_blocks: list[dict[str, Any]],
    *,
    preserve_block_content: bool,
) -> Any:
    """Rebuild multimodal message content from flattened text plus non-text blocks."""
    if not preserve_block_content:
        return flat_text

    if not non_text_blocks:
        if not flat_text:
            return []
        return [{"type": "text", "text": flat_text}]

    parts = flat_text.split(MULTIMODAL_PLACEHOLDER)
    rebuilt: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        if part:
            rebuilt.append({"type": "text", "text": part})
        if index < len(non_text_blocks):
            rebuilt.append(non_text_blocks[index])
    return rebuilt


def align_messages_for_thinking(
    messages: list[dict[str, Any]],
    *,
    thinking_mode: bool | None | str,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Apply LF-private `enable_thinking` rewriting to assistant messages.

    Returns:
        A pair of:
        - rewritten messages for tokenization
        - assistant message indices whose leading empty thought block should be
          excluded from supervision because LF would place it in the prompt.
    """
    rewritten_messages: list[dict[str, Any]] = []
    assistant_skip_think_prefix: set[int] = set()

    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            rewritten_messages.append(message)
            continue

        content_text, non_text_blocks, preserve_block_content = _flatten_message_content(message.get("content"))
        modified_content = content_text
        was_thinking_empty = False

        if thinking_mode is False:
            modified_content = remove_thought(content_text)
        elif thinking_mode == "non-empty":
            was_thinking_empty = is_thought_empty(content_text)
            if was_thinking_empty:
                modified_content = remove_thought(content_text)

        has_modified_thought = all(marker in modified_content for marker in THOUGHT_MARKERS)
        if not has_modified_thought:
            if thinking_mode is None:
                pass
            elif thinking_mode is False:
                modified_content = add_thought(modified_content)
                assistant_skip_think_prefix.add(message_index)
            elif thinking_mode == "non-empty":
                if was_thinking_empty:
                    modified_content = add_thought(modified_content)
                    assistant_skip_think_prefix.add(message_index)
            else:
                modified_content = add_thought(modified_content)

        rewritten_message = dict(message)
        rewritten_message["content"] = _rebuild_message_content(
            modified_content,
            non_text_blocks,
            preserve_block_content=preserve_block_content,
        )
        rewritten_messages.append(rewritten_message)

    return rewritten_messages, assistant_skip_think_prefix


def build_labels(
    apply_chat_template: Any,
    messages: list[dict[str, Any]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_length: int,
    ignore_index: int,
    assistant_skip_think_prefix: set[int] | None = None,
) -> torch.Tensor:
    """Build supervised labels for one tokenized conversation.

    Args:
        apply_chat_template: Bound ``processor.apply_chat_template`` callable.
        messages: Rendered conversation in HF chat format.
        input_ids: Token ids for the full conversation.
        attention_mask: Attention mask for the full conversation.
        max_length: Maximum tokenized length after truncation.
        ignore_index: Label value used to mask tokens out of the loss.
        assistant_skip_think_prefix: Assistant message indices whose leading
            empty thought block should be excluded from supervision.

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
        if assistant_skip_think_prefix is not None and message_index in assistant_skip_think_prefix and start < end:
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
    thinking_mode: bool | None | str = True,
) -> ModelInputs:
    """Validate one dataset row and convert it into training tensors.

    Args:
        sample: One raw row emitted by ``mvp_dataset`` from JSONL or parquet.
        processor: Hugging Face processor (or compatible object with
            ``apply_chat_template`` and ``__fingerprint__``).
        max_length: Maximum tokenized sequence length.
        image_placeholder: Placeholder token that marks image positions in text.
        ignore_index: Label value used to mask tokens out of the loss.
        thinking_mode: ``enable_thinking`` mode aligned to LF-private.

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
        rendered_messages, assistant_skip_think_prefix = align_messages_for_thinking(
            rendered_messages,
            thinking_mode=thinking_mode,
        )

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
            assistant_skip_think_prefix=assistant_skip_think_prefix,
        )
        if not torch.any(labels != ignore_index):
            raise ValueError("has no supervised assistant tokens after tokenization/truncation.")
    except (OSError, SyntaxError, ValueError) as exc:
        logger.warning(f"Skipping invalid sample {loc}: {exc}")
        return build_skipped_sample()
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


def lightweight_process_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    max_length: int,
    image_placeholder: str = IMAGE_PLACEHOLDER,
    ignore_index: int = -100,
    thinking_mode: bool | None | str = True,
) -> ModelInputs:
    """Build a fake sample whose token length matches the real multimodal prompt.

    This lightweight path is intended for pre-training accounting such as
    estimating packed sample counts before the real training preprocess runs.
    It skips vision tensor creation and only preserves the token-length
    behaviour of ``process_sample``, including LF-aligned thinking rewrites.
    """
    if not isinstance(sample, dict):
        raise ValueError(f"Expected a dictionary sample, got {type(sample).__name__}.")

    apply_chat_template = processor.apply_chat_template
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor must expose a tokenizer for lightweight preprocessing.")

    source_file = sample.get("__file__")
    index_in_file = sample.get("__index_in_file__")
    if not isinstance(source_file, str) or not source_file:
        raise ValueError("mvp_dataset samples must include a string `__file__` field.")
    if not isinstance(index_in_file, int) or index_in_file < 0:
        raise ValueError("mvp_dataset samples must include a non-negative integer `__index_in_file__` field.")

    source_file = Path(source_file).expanduser().resolve()
    row_number = index_in_file + 1
    loc = f"{source_file}:{row_number}"

    def resolve_image_size(image: str | bytes | dict[str, Any]) -> tuple[int, int]:
        """Read only image metadata needed to recover multimodal token length."""
        if isinstance(image, dict):
            image_bytes = image.get("bytes")
            if isinstance(image_bytes, (bytes, bytearray, memoryview)):
                return resolve_image_size(bytes(image_bytes))

            image_path = image.get("path")
            if isinstance(image_path, str) and image_path:
                return resolve_image_size(image_path)

            raise ValueError("contains an invalid image record.")
        if isinstance(image, bytes):
            # with Image.open(io.BytesIO(image)) as decoded:
            #     width, height = decoded.size
            width, height = imagesize.get(io.BytesIO(image))
            return height, width

        if not isinstance(image, str):
            raise ValueError(f"contains an invalid image value: {type(image).__name__}.")
        if not image:
            raise ValueError(f"contains an invalid image path: {image!r}")

        resolved = Path(image).expanduser()
        if not resolved.is_absolute():
            resolved = source_file.parent / resolved
        resolved = resolved.resolve()

        if not resolved.is_file():
            raise FileNotFoundError(f"references missing image: {resolved}")

        with Image.open(resolved) as decoded:
            width, height = decoded.size
        return height, width

    try:
        messages = sample.get("messages")
        if messages is None:
            messages = sample.get("conversations")
        images = sample.get("images", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("has invalid `messages`/`conversations`.")
        if not isinstance(images, list):
            raise ValueError("has invalid `images`.")

        image_sizes = [resolve_image_size(image) for image in images]
        image_iter = iter([f"__panguvl_fake_image_{index}__" for index in range(len(image_sizes))])
        rendered_messages = [process_message(msg, image_iter, image_placeholder=image_placeholder) for msg in messages]
        rendered_messages, _ = align_messages_for_thinking(
            rendered_messages,
            thinking_mode=thinking_mode,
        )

        unused = list(image_iter)
        if unused:
            raise ValueError(f"has {len(unused)} unused image(s).")

        prompt = apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(prompt, str):
            raise ValueError("processor.apply_chat_template must return a prompt string when tokenize=False.")

        expanded_prompt = prompt
        if image_sizes:
            image_processor = getattr(processor, "image_processor", None)
            image_token = getattr(processor, "image_token", None)
            if image_processor is None:
                raise ValueError("Processor must expose an image_processor for multimodal lightweight preprocessing.")
            if not callable(getattr(image_processor, "get_number_of_image_patches", None)):
                raise ValueError("Processor image_processor must expose get_number_of_image_patches.")
            if not isinstance(image_token, str) or not image_token:
                raise ValueError("Processor must expose a valid image token.")

            merge_size = getattr(image_processor, "merge_size", None)
            if not isinstance(merge_size, int) or merge_size <= 0:
                raise ValueError("Processor image_processor must expose a positive integer merge_size.")

            placeholder_token = "<|panguvl_image_token_placeholder|>"
            for height, width in image_sizes:
                num_image_patches = image_processor.get_number_of_image_patches(height, width, {})
                num_image_tokens = num_image_patches // (merge_size**2)
                expanded_prompt = expanded_prompt.replace(image_token, placeholder_token * num_image_tokens, 1)
            expanded_prompt = expanded_prompt.replace(placeholder_token, image_token)

        tokenizer_kwargs: dict[str, Any] = {
            "truncation": True,
            "max_length": max_length,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        bos_token = getattr(tokenizer, "bos_token", None)
        if isinstance(bos_token, str) and bos_token and expanded_prompt.startswith(bos_token):
            tokenizer_kwargs["add_special_tokens"] = False

        tokenized = tokenizer(expanded_prompt, **tokenizer_kwargs)
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
    except (OSError, SyntaxError, ValueError) as exc:
        logger.warning(f"Skipping invalid sample {loc}: {exc}")
        return build_skipped_sample()
    except Exception as exc:
        raise type(exc)(f"{loc} {exc}") from exc

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.full_like(input_ids, ignore_index),
    }


def build_dataset(
    config: Any,
    *,
    processor: Any,
    process_fn: Any = process_sample,
    resample: bool = True,
) -> Dataset:
    """Build the training dataset pipeline for the recipe.

    Args:
        config: Recipe config with dataset, runtime, and shuffle settings.
        processor: Hugging Face processor used during sample processing.
        process_fn: Function to process individual samples.
        resample: Whether to loop dataset shards indefinitely across rounds.
    Returns:
        An ``mvp_dataset.Dataset`` pipeline with parquet loading, processing, and
        sample-level shuffling.
    """
    set_logger(logger)
    dataset_path_value = config.data.train_path
    if dataset_path_value is None:
        raise ValueError("Missing `data.train_path` for the PanguVL recipe.")
    dataset_paths = resolve_dataset_shards(dataset_path_value)

    context = RuntimeContext.from_runtime(seed=int(config.seed))

    process_kwargs: dict[str, Any] = {
        "processor": processor,
        "max_length": int(config.data.max_seq_len),
    }
    if process_fn in {process_sample, lightweight_process_sample}:
        process_kwargs["thinking_mode"] = getattr(config.data, "enable_thinking", True)

    dataset = Dataset.from_source(
        "parquet",
        dataset_paths,
        context=context,
        resample=resample,
    ).map(partial(process_fn, **process_kwargs))
    dataset = dataset.assemble(build_invalid_sample_gate_assembler)

    if config.data.cache:
        cache_write_batch_size = int(getattr(config.data, "cache_write_batch_size", 32))
        configure_cache_write_batch_size(cache_write_batch_size)
        logger.info("PanguVL cache: using write batch size %d", cache_write_batch_size)
        dataset = dataset.cache(
            cache_dir=str(resolve_cache_dir(dataset_path_value, getattr(config.data, "cache_dir", None))),
            cache_num_workers=int(getattr(config.data, "cache_num_workers", 1)),
        )

    dataset = dataset.shuffle(buffer_size=config.data.shuffle_buffer)

    if config.data.packing:
        max_length = config.data.max_seq_len
        selection_strategy = config.data.packing_selection_strategy
        open_pack_limit = config.data.packing_open_pack_limit
        pack_buffer_size = config.data.packing_buffer_size

        dataset = dataset.assemble(
            partial(
                build_packed_sample_assembler,
                max_length=max_length,
                selection_strategy=selection_strategy,
                open_pack_limit=open_pack_limit,
                pack_buffer_size=pack_buffer_size,
            )
        )
        if config.data.shuffle_on_packs:
            dataset = dataset.shuffle(buffer_size=int(config.data.shuffle_on_packs_buffer))

    return dataset
