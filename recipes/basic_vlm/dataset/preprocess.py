import io
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from torchvision.transforms import InterpolationMode

from mvp_engine.utils.log import simple_info

from ..guards.data import build_empty_sample

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
IMAGE_PLACEHOLDER = "<image>"
MULTIMODAL_PLACEHOLDER = "<|mvp_multimodal_placeholder|>"
IMAGE_TOKEN_PLACEHOLDER = "<|mvp_image_placeholder|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
_VISION_TOKEN_IDS: set[int] = set()
_VISION_TOKEN_ID_TENSOR: torch.Tensor | None = None


def process_image(
    image: str | bytes | dict[str, Any] | Image.Image,
    *,
    image_root: Path | None = None,
) -> Image.Image:
    """Normalize one image input from one dataset row.

    Args:
        image: Image path, raw image bytes, PIL image, or a parquet image
            struct with ``{"bytes", "path"}`` fields.
        image_root: Optional base directory used for relative image paths.

    Returns:
        A decoded RGB PIL image.
    """
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            return process_image(bytes(image_bytes), image_root=image_root)

        image_path = image.get("path")
        if isinstance(image_path, str) and image_path:
            return process_image(image_path, image_root=image_root)

        raise ValueError("contains an invalid image record.")

    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()

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
    with Image.open(resolved) as opened:
        return opened.convert("RGB").copy()


def convert_images_to_pixel_values(
    sample: dict[str, Any] | list[dict[str, Any]],
    *,
    processor: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert resolved Basic VLM images into Qwen image-processor tensors.

    Packed groups may arrive as a list while image references are still
    unresolved. Each member is materialized independently before final packing.
    """
    if isinstance(sample, list):
        return [convert_images_to_pixel_values(s, processor=processor) for s in sample]

    try:
        images = sample.get("images", [])
        adjusted_sizes = sample.get("adjusted_image_size", [])
        if not images:
            sample.pop("images", None)
            sample.pop("adjusted_image_size", None)
            return sample

        def _load_resized_image_tensor(image: Any, target_size: list[int]) -> torch.Tensor:
            """Load one image and resize it to its precomputed smart-resize shape."""
            height, width = int(target_size[0]), int(target_size[1])
            pil_image = process_image(image)
            image_tensor = tvF.pil_to_tensor(pil_image)
            if tuple(image_tensor.shape[-2:]) != (height, width):
                image_tensor = tvF.resize(
                    image_tensor,
                    [height, width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
            return image_tensor

        resized_images = [
            _load_resized_image_tensor(image, target_size)
            for image, target_size in zip(images, adjusted_sizes, strict=True)
        ]

        image_inputs = processor.image_processor(
            images=resized_images,
            do_resize=False,
            return_tensors="pt",
        )
        sample.pop("images", None)
        sample.pop("adjusted_image_size", None)
        sample["pixel_values"] = image_inputs["pixel_values"]
        sample["image_grid_thw"] = image_inputs["image_grid_thw"]
        return sample
    except Exception as exc:
        simple_info(exc, level="debug")
        return build_empty_sample()


def _resolve_image_processor_config(processor: Any) -> tuple[Any, int, int, int, int]:
    """Resolve Qwen image processor geometry needed for token estimation and resizing."""
    image_processor = getattr(processor, "image_processor", processor)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("Processor image processor must expose a positive integer `patch_size`.")
    if not isinstance(merge_size, int) or merge_size <= 0:
        raise ValueError("Processor image processor must expose a positive integer `merge_size`.")

    image_processor_size = getattr(image_processor, "size", {})
    min_pixels = getattr(processor, "min_image_size", None)
    max_pixels = getattr(processor, "max_image_size", None)
    if min_pixels is None and isinstance(image_processor_size, dict):
        min_pixels = image_processor_size.get("shortest_edge")
    if max_pixels is None and isinstance(image_processor_size, dict):
        max_pixels = image_processor_size.get("longest_edge")
    if not isinstance(min_pixels, int) or min_pixels <= 0:
        raise ValueError("Processor image processor must expose a positive integer min pixel budget.")
    if not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ValueError("Processor image processor must expose a positive integer max pixel budget.")

    return image_processor, patch_size, merge_size, min_pixels, max_pixels


def _smart_resize_image_size(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int] | None:
    """Resize an image size with Qwen2-VL smart-resize semantics."""
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return int(resized_height), int(resized_width)
    except Exception:
        pass

    try:
        height = int(height)
        width = int(width)
        if height <= 0 or width <= 0 or factor <= 0:
            return None
        if max(height, width) / min(height, width) > 200:
            return None

        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        if resized_height * resized_width > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            resized_height = max(factor, math.floor(height / beta / factor) * factor)
            resized_width = max(factor, math.floor(width / beta / factor) * factor)
        elif resized_height * resized_width < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
    except Exception:
        return None

    return int(resized_height), int(resized_width)


def _estimate_image_tokens(
    image_processor: Any,
    *,
    height: int,
    width: int,
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
) -> int | None:
    """Estimate Qwen2-VL language-side image tokens for one raw image size."""
    get_number_of_image_patches = getattr(image_processor, "get_number_of_image_patches", None)

    try:
        if callable(get_number_of_image_patches):
            num_patches = get_number_of_image_patches(
                height,
                width,
                {
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "patch_size": patch_size,
                    "merge_size": merge_size,
                },
            )
            return int(num_patches // (merge_size**2))
    except Exception:
        pass

    resized_size = _smart_resize_image_size(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if resized_size is None:
        return None

    resized_height, resized_width = resized_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    return int((grid_h * grid_w) // (merge_size**2))


def _apply_thinking_mode(
    messages: list[dict[str, Any]],
    *,
    thinking_mode: bool | None | str,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Rewrite assistant thinking blocks.

    ``False`` and ``"non-empty"`` may still insert an empty thinking block for
    prompt compatibility. The returned index set marks those inserted empty
    blocks so label construction can skip them later.
    """
    rewritten_messages: list[dict[str, Any]] = []
    assistant_skip_think_prefix: set[int] = set()

    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            rewritten_messages.append(message)
            continue

        content = message.get("content")
        non_text_blocks: list[dict[str, Any]] = []
        preserve_blocks = isinstance(content, list)

        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                elif isinstance(block, dict):
                    text_parts.append(MULTIMODAL_PLACEHOLDER)
                    non_text_blocks.append(block)
            content_text = "".join(text_parts)
        else:
            content_text = ""

        thought_match = THOUGHT_PATTERN.search(content_text)
        thought_is_empty = thought_match is None or not thought_match.group(1).strip()
        modified_text = content_text

        if thinking_mode is False:
            modified_text = THOUGHT_PATTERN.sub("", content_text).lstrip("\n")
        elif thinking_mode == "non-empty" and thought_is_empty:
            modified_text = THOUGHT_PATTERN.sub("", content_text).lstrip("\n")

        has_thought_block = all(marker in modified_text for marker in THOUGHT_MARKERS)
        should_add_empty_thought = False
        should_skip_added_thought = False

        if not has_thought_block:
            if thinking_mode is False:
                should_add_empty_thought = True
                should_skip_added_thought = True
            elif thinking_mode == "non-empty":
                should_add_empty_thought = thought_is_empty
                should_skip_added_thought = thought_is_empty
            elif thinking_mode is not None:
                should_add_empty_thought = True

        if should_add_empty_thought:
            modified_text = f"{THOUGHT_PREFIX}{THOUGHT_SUFFIX}{modified_text}"
            if should_skip_added_thought:
                assistant_skip_think_prefix.add(message_index)

        rewritten_message = dict(message)
        if preserve_blocks:
            if non_text_blocks:
                parts = modified_text.split(MULTIMODAL_PLACEHOLDER)
                rebuilt_blocks: list[dict[str, Any]] = []
                for part_index, part in enumerate(parts):
                    if part:
                        rebuilt_blocks.append({"type": "text", "text": part})
                    if part_index < len(non_text_blocks):
                        rebuilt_blocks.append(non_text_blocks[part_index])
                rewritten_message["content"] = rebuilt_blocks
            else:
                rewritten_message["content"] = [{"type": "text", "text": modified_text}] if modified_text else []
        else:
            rewritten_message["content"] = modified_text

        rewritten_messages.append(rewritten_message)

    return rewritten_messages, assistant_skip_think_prefix


def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
    """Normalize a raw conversation message to the Basic VLM chat schema.

    The preprocessor accepts both already-normalized messages with
    ``{"role": ..., "content": ...}`` fields and common source-dataset
    messages with ``{"from": ..., "value": ...}`` fields. Roles are mapped
    through ``ROLE_MAP`` so source aliases such as ``"human"`` and ``"gpt"``
    become the canonical ``"user"`` and ``"assistant"`` roles.

    Args:
        message: One raw message dictionary from a sample's conversation list.

    Returns:
        A dictionary with canonical ``role`` and string ``content`` keys.

    Raises:
        ValueError: If the role is missing, unknown, or the selected content
            field is not a string.
    """
    role = message.get("role")
    content = message.get("content")
    if isinstance(role, str) and isinstance(content, str) and role:
        normalized_role = ROLE_MAP.get(role)
        if normalized_role is None:
            raise ValueError(f"contains an invalid role: {role!r}")
        return {"role": normalized_role, "content": content}

    source_role = message.get("from")
    source_content = message.get("value")
    normalized_role = ROLE_MAP.get(source_role)
    if normalized_role is None:
        raise ValueError(f"contains an invalid role: {source_role!r}")
    if not isinstance(source_content, str):
        raise ValueError("contains non-string content.")
    return {"role": normalized_role, "content": source_content}


def _process_message(
    message: dict[str, Any],
    images: Iterable[str],
    *,
    image_placeholder: str,
) -> dict[str, Any]:
    """Convert one source message into HF chat content blocks.

    Args:
        message: Source message in recipe JSONL format or parquet Open-Bee format.
        images: Iterable of raw images from the dataset for ``<image>`` placeholders.
        image_placeholder: Placeholder token that marks image positions in text.

    Returns:
        A Hugging Face chat-format message with ``text`` and ``image`` blocks.
    """
    normalized_message = _normalize_message(message)
    role = normalized_message["role"]
    content = normalized_message["content"]

    blocks: list[dict[str, Any]] = []
    segments = content.split(image_placeholder)
    for i, segment in enumerate(segments):
        if segment:
            blocks.append({"type": "text", "text": segment})
        if i < len(segments) - 1:
            try:
                image_value = next(images)
                blocks.append({"type": "image", "image": image_value})
            except StopIteration as exc:
                raise ValueError("has more image placeholders than images.") from exc

    rendered_message: dict[str, Any] = {"role": role, "content": blocks}
    if role == "assistant" and "tool_calls" in message:
        rendered_message["tool_calls"] = message["tool_calls"]
    return rendered_message


def _normalize_image_size(size_entry: Any) -> list[int]:
    """Parse Basic VLM image metadata into ``[height, width]``.

    The input may be a dict with ``{"height", "width"}`` fields or a list/tuple
    with width and height as the first two elements.
    """
    if isinstance(size_entry, dict):
        width = size_entry.get("width")
        height = size_entry.get("height")
    elif isinstance(size_entry, (list, tuple)) and len(size_entry) >= 2:
        width = size_entry[0]
        height = size_entry[1]
    else:
        raise ValueError(f"contains invalid image size metadata: {size_entry!r}")

    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(f"contains invalid image size metadata: {size_entry!r}")
    return [int(height), int(width)]


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """Allocate the remaining token budget between source and target text."""
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    return min(max_source_len, source_len), new_target_len


def process_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    max_length: int,
    image_placeholder: str = IMAGE_PLACEHOLDER,
    ignore_index: int = -100,
    thinking_mode: bool | None | str = True,
):
    """Convert one dataset row into training inputs.

    Args:
        sample: One raw row emitted by the raw dataset.
        processor: Hugging Face processor (or compatible object with
                   ``apply_chat_template``).
        max_length: Maximum tokenized sequence length. Any input longer than this will be truncated.
        image_placeholder: Placeholder token that marks image positions in text.
        ignore_index: Label value used to mask tokens out of the loss.
        thinking_mode: The policy for handling <think>...</think>.

    Returns:
        A processed sample containing token tensors and optional vision tensors.
    """
    try:
        messages = sample.get("messages") or sample.get("conversations")
        images = list(sample.get("images", []))
        raw_image_sizes = sample.get("img_size", []) or sample.get("image_size", [])
        image_sizes = [_normalize_image_size(size) for size in raw_image_sizes]

        # 1. Render the messages into a chat format with interleaved text and image blocks.
        images_iter = iter(images)
        rendered_messages = [
            _process_message(
                message=message,
                images=images_iter,
                image_placeholder=image_placeholder,
            )
            for message in messages
        ]
        if list(images_iter):
            raise ValueError("has more images than image placeholders.")

        # 2. Rewrite assistant <think>...</think> blocks before tokenization.
        rendered_messages, assistant_skip_think_prefix = _apply_thinking_mode(
            rendered_messages,
            thinking_mode=thinking_mode,
        )

        # 3. Smart-resize image metadata once. If text truncation would cut vision tokens later,
        # skip the sample rather than silently corrupting image/text alignment.
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Processor does not have a tokenizer attribute for tokenization.")

        image_processor, patch_size, merge_size, min_pixels, max_pixels = _resolve_image_processor_config(processor)
        factor = patch_size * merge_size
        adjusted_image_sizes: list[list[int]] = []
        image_token_counts: list[int] = []
        for size in image_sizes:
            height, width = int(size[0]), int(size[1])
            resized_size = _smart_resize_image_size(
                height,
                width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if resized_size is None:
                raise ValueError(f"cannot smart-resize image size {height}x{width}.")

            resized_height, resized_width = resized_size
            adjusted_image_sizes.append([int(resized_height), int(resized_width)])
            token_count = _estimate_image_tokens(
                image_processor,
                height=int(resized_height),
                width=int(resized_width),
                patch_size=patch_size,
                merge_size=merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if token_count is None or token_count <= 0:
                raise ValueError(f"cannot estimate image token count after resize: {resized_height}x{resized_width}.")
            image_token_counts.append(int(token_count))

        sample["adjusted_image_size"] = adjusted_image_sizes

        # 4. Build multiturn source/target pairs through the processor chat template.
        image_token = getattr(processor, "image_token", DEFAULT_IMAGE_TOKEN)
        if not isinstance(image_token, str) or not image_token:
            raise ValueError("Processor must expose a valid image token.")

        image_cursor = 0

        def _expand_image_placeholders(text: str) -> str:
            """Expand each image placeholder to the estimated number of image tokens."""
            nonlocal image_cursor

            placeholder = IMAGE_TOKEN_PLACEHOLDER if IMAGE_TOKEN_PLACEHOLDER in text else image_token
            parts = text.split(placeholder)
            if len(parts) == 1:
                return text

            expanded_parts = [parts[0]]
            for part in parts[1:]:
                token_count = image_token_counts[image_cursor]
                image_cursor += 1
                expanded_parts.append(image_token * token_count)
                expanded_parts.append(part)
            return "".join(expanded_parts)

        turns: list[tuple[str, str]] = []
        leading_system_messages: list[dict[str, Any]] = []
        used_image_count = 0
        message_index = 0
        while message_index < len(rendered_messages) and rendered_messages[message_index].get("role") == "system":
            leading_system_messages.append(rendered_messages[message_index])
            message_index += 1

        empty_thought = f"{THOUGHT_PREFIX}{THOUGHT_SUFFIX}"
        while message_index < len(rendered_messages):
            user_message = rendered_messages[message_index]
            if user_message.get("role") != "user":
                raise ValueError("conversation must contain user/assistant turn pairs after optional system messages.")
            if message_index + 1 >= len(rendered_messages):
                break

            assistant_message = rendered_messages[message_index + 1]
            if assistant_message.get("role") != "assistant":
                raise ValueError("conversation must contain user/assistant turn pairs after optional system messages.")

            source_messages = leading_system_messages + [user_message]
            full_messages = source_messages + [assistant_message]
            used_image_count += sum(
                1
                for message in full_messages
                for block in message.get("content", [])
                if isinstance(block, dict) and block.get("type") == "image"
            )
            raw_source_text = processor.apply_chat_template(
                source_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            raw_full_text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not raw_full_text.startswith(raw_source_text):
                raise ValueError("processor chat template does not preserve source prefix for assistant target split.")

            source_text = _expand_image_placeholders(raw_source_text)
            target_text = _expand_image_placeholders(raw_full_text[len(raw_source_text) :])
            if message_index + 1 in assistant_skip_think_prefix and target_text.startswith(empty_thought):
                source_text += empty_thought
                target_text = target_text[len(empty_thought) :]

            turns.append((source_text, target_text))
            leading_system_messages = []
            message_index += 2

        if image_cursor != used_image_count:
            raise ValueError("image size metadata does not match rendered image placeholders.")
        sample["images"] = images[:used_image_count]
        sample["adjusted_image_size"] = adjusted_image_sizes[:used_image_count]

        # 5. Tokenize and truncate each turn independently under a shared max-length budget.
        global _VISION_TOKEN_ID_TENSOR
        if not _VISION_TOKEN_IDS:
            for token in (VISION_START_TOKEN, VISION_END_TOKEN, image_token):
                _VISION_TOKEN_IDS.update(tokenizer(token, add_special_tokens=False)["input_ids"])
            _VISION_TOKEN_ID_TENSOR = torch.tensor(sorted(_VISION_TOKEN_IDS), dtype=torch.long)

        def _will_cut_vision_tokens(token_ids: list[int], keep_len: int) -> bool:
            """Return whether truncation would cut away any vision special tokens."""
            return any(token_id in _VISION_TOKEN_IDS for token_id in token_ids[keep_len:])

        input_ids_list: list[int] = []
        labels_list: list[int] = []
        total_length = 0
        for source_text, target_text in turns:
            if total_length >= max_length:
                break

            source_ids = tokenizer(source_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
            source_len, target_len = _infer_seqlen(len(source_ids), len(target_ids), max_length - total_length)
            if _will_cut_vision_tokens(source_ids, source_len) or _will_cut_vision_tokens(target_ids, target_len):
                raise ValueError("wrong cutoff.")

            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            input_ids_list.extend(source_ids)
            input_ids_list.extend(target_ids)
            labels_list.extend([ignore_index] * len(source_ids))
            labels_list.extend(target_ids)
            total_length += len(source_ids) + len(target_ids)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)
        labels[torch.isin(input_ids, _VISION_TOKEN_ID_TENSOR)] = ignore_index
        if not torch.any(labels != ignore_index):
            raise ValueError("has no supervised assistant tokens after tokenization/truncation.")

        attention_mask = torch.ones_like(input_ids)
        sample.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

        return sample

    except Exception as exc:
        simple_info(exc, level="debug")
        return build_empty_sample()
