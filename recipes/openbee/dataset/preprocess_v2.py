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
MULTIMODAL_PLACEHOLDER = "<|openbee_multimodal_placeholder|>"


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


def convert_images_to_pixel_values(sample: dict[str, Any], *, processor: Any) -> dict[str, Any]:
    """Convert resolved OpenBee images into Qwen image-processor tensors."""
    images = sample.pop("images", [])
    adjusted_sizes = sample.pop("adjusted_image_size", [])
    if not images:
        return sample

    def _load_resized_image_tensor(image: Any, target_size: list[int]) -> torch.Tensor:
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
    sample["pixel_values"] = image_inputs["pixel_values"]
    sample["image_grid_thw"] = image_inputs["image_grid_thw"]
    return sample


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
    """Normalize a raw conversation message to the OpenBee chat schema.

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
    """Parse OpenBee image metadata into ``[height, width]``.
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
        sample: One raw row emitted by ``mvp_dataset``.
        processor: Hugging Face processor (or compatible object with
                   ``apply_chat_template`` and ``__fingerprint__``).
        max_length: Maximum tokenized sequence length. Any input longer than this will be truncated.
        image_placeholder: Placeholder token that marks image positions in text.
        ignore_index: Label value used to mask tokens out of the loss.
        thinking_mode: The policy for handling <think>...</think>.

    Returns:
        A processed sample containing token tensors and optional vision tensors.
    """
    try:
        messages = sample.get("messages") or sample.get("conversations")
        images = list(sample["images"])
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
        _ = assistant_skip_think_prefix

        # 3. Convert messages into prompt and tokens.
        prompt = processor.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if tokenizer := getattr(processor, "tokenizer", None):
            tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
            )
            text_input_ids = tokenized["input_ids"]
        else:
            raise ValueError("Processor does not have a tokenizer attribute for tokenization.")

        image_processor, patch_size, merge_size, min_pixels, max_pixels = _resolve_image_processor_config(processor)

        # 4. Calculate the original estimated image tokens.
        estimated_image_tokens = [
            _estimate_image_tokens(
                image_processor,
                height=size[0],
                width=size[1],
                patch_size=patch_size,
                merge_size=merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            for size in image_sizes
        ]
        invalid_token_indices = [
            index for index, token_count in enumerate(estimated_image_tokens) if token_count is None or token_count <= 0
        ]
        if invalid_token_indices:
            raise ValueError(
                f"cannot estimate a positive image token count for image index(es): {invalid_token_indices}."
            )
        total_estimated_image_tokens = sum(int(token_count) for token_count in estimated_image_tokens)

        # 5. Recalculate the image size budget based on the text token number and the max_length.
        # ignore the existing image tokens in the input_ids for simplicity.
        # we * 0.95 to reserve some budget for special tokens and potential underestimation of token counts.
        image_token_budget = (max_length - len(text_input_ids)) * 0.95
        single_image_minimal_token_budget = _estimate_image_tokens(
            image_processor,
            height=min_pixels**0.5,
            width=min_pixels**0.5,
            patch_size=patch_size,
            merge_size=merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if single_image_minimal_token_budget is None or single_image_minimal_token_budget <= 0:
            raise ValueError("cannot estimate the minimal image token budget.")
        if image_token_budget <= single_image_minimal_token_budget * len(image_sizes):
            raise ValueError("The text tokens have already exceeded the max_length, no budget left for images.")

        # 6. Calculate the new target image size for each image based on the estimated tokens and the budget.
        adjusted_image_sizes: list[list[int]] = []
        scale_factor = (
            math.sqrt(image_token_budget / total_estimated_image_tokens) if total_estimated_image_tokens > 0 else 1.0
        )
        if scale_factor < 1.0:
            for size in image_sizes:
                new_height = max(1, int(size[0] * scale_factor))
                new_width = max(1, int(size[1] * scale_factor))
                adjusted_image_sizes.append([new_height, new_width])
        else:
            adjusted_image_sizes = image_sizes

        # 7. Smart resize the new image sizes to be compatible with the model's patch and merge sizes.
        factor = patch_size * merge_size
        smart_adjusted_image_sizes: list[list[int]] = []
        for size in adjusted_image_sizes:
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
            smart_adjusted_image_sizes.append([int(resized_height), int(resized_width)])
        adjusted_image_sizes = smart_adjusted_image_sizes
        sample["adjusted_image_size"] = adjusted_image_sizes

        # 8. Expand the image token placeholders in the prompt.
        image_token = getattr(processor, "image_token", None)
        if not isinstance(image_token, str) or not image_token:
            raise ValueError("Processor must expose a valid image token.")

        expanded_prompt = prompt
        image_token_placeholder = f"<image_{MULTIMODAL_PLACEHOLDER}_tokens>"
        prompt_image_placeholder = (
            image_token_placeholder if image_token_placeholder in expanded_prompt else image_token
        )
        image_token_placeholder_pattern = re.escape(prompt_image_placeholder)
        temporary_image_token = "<|openbee_image_token_placeholder|>"
        for _, adj_size in zip(estimated_image_tokens, adjusted_image_sizes, strict=True):
            new_est_tokens = _estimate_image_tokens(
                image_processor,
                height=adj_size[0],
                width=adj_size[1],
                patch_size=patch_size,
                merge_size=merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if new_est_tokens is None or new_est_tokens <= 0:
                raise ValueError(f"cannot estimate image token count after resize: {adj_size!r}.")
            token_placeholder = temporary_image_token * int(new_est_tokens)

            expanded_prompt, replacement_count = re.subn(
                image_token_placeholder_pattern,
                lambda _match: token_placeholder,
                expanded_prompt,
                count=1,
            )
            if replacement_count != 1:
                raise ValueError("prompt has fewer image token placeholders than image sizes.")
        prompt = expanded_prompt.replace(temporary_image_token, image_token)

        # 9. Tokenize the final prompt and build assistant-only labels from Qwen chat spans.
        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        def _token_count(text: str) -> int:
            counted = tokenizer(
                text,
                return_tensors="pt",
                return_attention_mask=False,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            return int(counted["input_ids"].shape[-1])

        labels = torch.full_like(input_ids, ignore_index)
        valid_length = int(attention_mask.sum().item())
        assistant_header = "<|im_start|>assistant\n"
        assistant_end = "<|im_end|>\n"
        assistant_message_indices = [
            index for index, message in enumerate(rendered_messages) if message.get("role") == "assistant"
        ]

        search_pos = 0
        assistant_span_index = 0
        while True:
            header_start = prompt.find(assistant_header, search_pos)
            if header_start < 0:
                break

            start_char = header_start + len(assistant_header)
            end_start = prompt.find(assistant_end, start_char)
            if end_start < 0:
                raise ValueError("assistant span is missing the Qwen end marker.")
            end_char = end_start + len(assistant_end)

            if assistant_span_index < len(assistant_message_indices):
                message_index = assistant_message_indices[assistant_span_index]
                empty_thought = f"{THOUGHT_PREFIX}{THOUGHT_SUFFIX}"
                if message_index in assistant_skip_think_prefix and prompt.startswith(empty_thought, start_char):
                    start_char += len(empty_thought)

            start = min(_token_count(prompt[:start_char]), valid_length)
            end = min(_token_count(prompt[:end_char]), valid_length)
            if start < end:
                labels[start:end] = input_ids[start:end]

            search_pos = end_char
            assistant_span_index += 1

        if not torch.any(labels != ignore_index):
            raise ValueError("has no supervised assistant tokens after tokenization/truncation.")

        sample.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

        return sample

    except Exception as exc:
        simple_info(
            f"Skipping sample with error: {sample}\n{exc}",
            level="warning",
        )
        return build_empty_sample()
