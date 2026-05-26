"""Text preprocessing utilities for the Qwen3 LM recipe."""

from __future__ import annotations

import re
from typing import Any

import torch

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


def _apply_thinking_mode(
    messages: list[dict[str, Any]],
    *,
    thinking_mode: bool | None | str,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Rewrite assistant thinking blocks according to the recipe policy."""
    rewritten_messages: list[dict[str, Any]] = []
    assistant_skip_think_prefix: set[int] = set()

    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            rewritten_messages.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, str):
            content = str(content or "")

        thought_match = THOUGHT_PATTERN.search(content)
        thought_is_empty = thought_match is None or not thought_match.group(1).strip()
        modified_text = content

        if thinking_mode is False:
            modified_text = THOUGHT_PATTERN.sub("", content).lstrip("\n")
        elif thinking_mode == "non-empty" and thought_is_empty:
            modified_text = THOUGHT_PATTERN.sub("", content).lstrip("\n")

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
        rewritten_message["content"] = modified_text
        rewritten_messages.append(rewritten_message)

    return rewritten_messages, assistant_skip_think_prefix


def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
    """Normalize one raw message to canonical Qwen chat roles."""
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


def _normalize_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Return normalized chat messages from supported raw sample shapes."""
    messages = sample.get("messages") or sample.get("conversations")
    if messages is None and "prompt" in sample and "response" in sample:
        messages = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]
    if not isinstance(messages, list) or not messages:
        raise ValueError("contains missing or invalid messages.")
    return [_normalize_message(message) for message in messages]


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    thinking_mode: bool | None | str,
) -> str:
    """Render chat text while passing Qwen3 thinking kwargs when supported."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if thinking_mode is not None:
        kwargs["enable_thinking"] = thinking_mode is not False

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


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


def _tensorize_pretokenized(
    sample: dict[str, Any],
    *,
    max_length: int,
    ignore_index: int,
) -> dict[str, torch.Tensor]:
    """Convert pre-tokenized rows into model inputs."""
    input_ids = torch.tensor(sample["input_ids"][:max_length], dtype=torch.long)
    labels_value = sample.get("labels")
    if labels_value is None:
        labels = input_ids.clone()
    else:
        labels = torch.tensor(labels_value[:max_length], dtype=torch.long)
    attention_value = sample.get("attention_mask")
    if attention_value is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = torch.tensor(attention_value[:max_length], dtype=torch.long)

    if labels.numel() != input_ids.numel() or attention_mask.numel() != input_ids.numel():
        raise ValueError("pre-tokenized input_ids, labels, and attention_mask must have matching lengths.")
    if not torch.any(labels != ignore_index):
        raise ValueError("has no supervised tokens after pre-tokenized truncation.")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def process_sample(
    sample: dict[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
    ignore_index: int = -100,
    thinking_mode: bool | None | str = "non-empty",
) -> dict[str, Any]:
    """Convert one dataset row into Qwen3 LM training inputs."""
    try:
        if isinstance(sample.get("input_ids"), list):
            return _tensorize_pretokenized(sample, max_length=max_length, ignore_index=ignore_index)

        rendered_messages, assistant_skip_think_prefix = _apply_thinking_mode(
            _normalize_messages(sample),
            thinking_mode=thinking_mode,
        )

        turns: list[tuple[str, str]] = []
        leading_system_messages: list[dict[str, str]] = []
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
            raw_source_text = _apply_chat_template(
                tokenizer,
                source_messages,
                add_generation_prompt=True,
                thinking_mode=thinking_mode,
            )
            raw_full_text = _apply_chat_template(
                tokenizer,
                full_messages,
                add_generation_prompt=False,
                thinking_mode=thinking_mode,
            )
            if not raw_full_text.startswith(raw_source_text):
                raise ValueError("tokenizer chat template does not preserve source prefix for assistant target split.")

            source_text = raw_source_text
            target_text = raw_full_text[len(raw_source_text) :]
            if message_index + 1 in assistant_skip_think_prefix and target_text.startswith(empty_thought):
                source_text += empty_thought
                target_text = target_text[len(empty_thought) :]

            turns.append((source_text, target_text))
            leading_system_messages = []
            message_index += 2

        input_ids_list: list[int] = []
        labels_list: list[int] = []
        total_length = 0
        for source_text, target_text in turns:
            if total_length >= max_length:
                break

            source_ids = tokenizer(source_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
            source_len, target_len = _infer_seqlen(len(source_ids), len(target_ids), max_length - total_length)

            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            input_ids_list.extend(source_ids)
            input_ids_list.extend(target_ids)
            labels_list.extend([ignore_index] * len(source_ids))
            labels_list.extend(target_ids)
            total_length += len(source_ids) + len(target_ids)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)
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
