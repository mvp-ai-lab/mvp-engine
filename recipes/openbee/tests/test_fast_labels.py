"""Fast label route coverage for the OpenBee dataset processor."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

import recipes.openbee.dataset.dataset as dataset_module
from recipes.openbee.dataset.dataset import (
    RECOVERABLE_SAMPLE_ERRORS,
    _compute_dynamic_retry_image_sizes,
    _replace_rendered_images,
    _resize_images_for_retry,
    _resolve_image_size_for_log,
    _tokenize_rendered_messages,
    align_messages_for_thinking,
    build_labels,
    process_image,
    process_message,
    process_sample,
)
from recipes.openbee.dataset.processor import build_qwen3_vl_processor

DEFAULT_MAX_LENGTH = 8192


@pytest.fixture(scope="session")
def processor() -> Any:
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "pretrained/Qwen3-VL-8B-Base-woDS-stage2",
        project_root / "pretrained/Qwen3-VL-8B-Base-woDS-stage1",
        project_root / "recipes/openbee/pretrained/Qwen3-VL-8B-Instruct",
    ]
    for candidate in candidates:
        if (candidate / "tokenizer_config.json").is_file() and (candidate / "preprocessor_config.json").is_file():
            return build_qwen3_vl_processor(
                SimpleNamespace(
                    pretrained_model_name_or_path=str(candidate),
                    image_max_pixels=5_062_500,
                )
            )
    pytest.skip("No local Qwen3-VL processor checkpoint is available.")


def _image_bytes(width: int, height: int, color: tuple[int, int, int] = (80, 120, 160)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pil_image(width: int, height: int, color: tuple[int, int, int] = (120, 70, 30)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _sample(
    messages: list[dict[str, Any]],
    *,
    images: list[Any] | None = None,
    img_size: list[Any] | None = None,
    file_name: str = "/tmp/openbee_fast_labels.jsonl",
) -> dict[str, Any]:
    sample = {
        "__file__": file_name,
        "__index_in_file__": 0,
        "messages": messages,
        "images": images or [],
    }
    if img_size is not None:
        sample["img_size"] = img_size
    return sample


def _conversation_sample(
    conversations: list[dict[str, Any]],
    *,
    images: list[Any] | None = None,
    img_size: list[Any] | None = None,
) -> dict[str, Any]:
    sample = _sample([], images=images, img_size=img_size)
    sample.pop("messages")
    sample["conversations"] = conversations
    return sample


def _old_reference_process_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    image_placeholder: str = "<image>",
    ignore_index: int = -100,
    thinking_mode: bool | None | str = True,
) -> dict[str, torch.Tensor]:
    apply_chat_template = processor.apply_chat_template
    source_file = Path(sample["__file__"]).expanduser().resolve()

    messages = sample.get("messages")
    if messages is None:
        messages = sample.get("conversations")
    images = sample.get("images", [])

    resolved_images = [process_image(image, image_root=source_file.parent) for image in images]
    image_iter = iter(resolved_images)
    rendered_messages = [process_message(msg, image_iter, image_placeholder=image_placeholder) for msg in messages]
    rendered_messages, assistant_skip_think_prefix = align_messages_for_thinking(
        rendered_messages,
        thinking_mode=thinking_mode,
    )
    unused = list(image_iter)
    if unused:
        raise ValueError(f"has {len(unused)} unused image(s).")

    label_messages = rendered_messages
    try:
        model_inputs = _tokenize_rendered_messages(
            apply_chat_template,
            rendered_messages,
            max_length=max_length,
        )
    except RECOVERABLE_SAMPLE_ERRORS as exc:
        err_message = str(exc)
        if not ("Mismatch in `image` token count" in err_message and "truncation" in err_message):
            raise

        image_sizes = [_resolve_image_size_for_log(image, image_root=source_file.parent) for image in resolved_images]
        retry_plan = _compute_dynamic_retry_image_sizes(
            apply_chat_template=apply_chat_template,
            processor=processor,
            rendered_messages=rendered_messages,
            image_sizes=image_sizes,
            max_length=max_length,
        )
        if retry_plan is None:
            raise

        target_sizes, _, _, _ = retry_plan
        retry_images = _resize_images_for_retry(
            resolved_images,
            target_sizes=target_sizes,
            image_root=source_file.parent,
        )
        label_messages = _replace_rendered_images(rendered_messages, retry_images)
        model_inputs = _tokenize_rendered_messages(
            apply_chat_template,
            label_messages,
            max_length=max_length,
        )

    input_ids = model_inputs["input_ids"][0]
    attention_mask = model_inputs["attention_mask"][0]
    labels = build_labels(
        apply_chat_template,
        label_messages,
        input_ids,
        attention_mask,
        max_length=max_length,
        ignore_index=ignore_index,
        assistant_skip_think_prefix=assistant_skip_think_prefix,
    )

    processed = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if "pixel_values" in model_inputs:
        processed["pixel_values"] = model_inputs["pixel_values"]
    if "image_grid_thw" in model_inputs:
        processed["image_grid_thw"] = model_inputs["image_grid_thw"]
    return processed


def _assert_fast_matches_old(
    sample: dict[str, Any],
    *,
    processor: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    thinking_mode: bool | None | str = True,
) -> dict[str, torch.Tensor]:
    fast = process_sample(sample, processor=processor, max_length=max_length, thinking_mode=thinking_mode)
    old = _old_reference_process_sample(
        sample,
        processor=processor,
        max_length=max_length,
        thinking_mode=thinking_mode,
    )

    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(fast[key], old[key]), key
    for key in ("pixel_values", "image_grid_thw"):
        assert key in fast or key not in old
        if key in fast:
            assert torch.equal(fast[key], old[key]), key
    assert torch.any(fast["labels"] != -100)
    return fast


def test_fast_labels_match_old_route_for_text_and_tool_calls(processor: Any) -> None:
    sample = _sample(
        [
            {"role": "system", "content": "你是一个严谨的助手。"},
            {"role": "user", "content": "先回答一个文本问题。"},
            {"role": "assistant", "content": "第一轮回答。"},
            {"role": "user", "content": "需要工具时也要保持格式。"},
            {
                "role": "assistant",
                "content": "我会调用工具。",
                "tool_calls": [{"name": "lookup", "arguments": {"query": "openbee label"}}],
            },
        ]
    )
    _assert_fast_matches_old(sample, processor=processor)


def test_fast_labels_match_old_route_for_conversation_schema_and_images(processor: Any) -> None:
    sample = _conversation_sample(
        [
            {"from": "system", "value": "系统提示。"},
            {"from": "human", "value": "<image>开头有图，中间有中文 and English."},
            {"from": "gpt", "value": "<think>\n非空思考\n</think>\n\n第一张图回答。"},
            {"from": "tool", "value": "工具返回文本。"},
            {"from": "human", "value": "连续图像：<image><image>，最后补一句。"},
            {"from": "gpt", "value": "第二轮回答。"},
        ],
        images=[
            _image_bytes(8, 8),
            {"bytes": _image_bytes(37, 53, (20, 90, 150))},
            _pil_image(21, 180),
        ],
        img_size=[
            {"width": 8, "height": 8},
            {"width": 37, "height": 53},
            [21, 180],
        ],
    )
    _assert_fast_matches_old(sample, processor=processor)


@pytest.mark.parametrize(
    ("content", "images"),
    [
        ("<image>图像在开头。", [_image_bytes(31, 47)]),
        ("图像在中间 <image> with English tail.", [{"bytes": _image_bytes(65, 33)}]),
        ("连续占位 <image><image> 后面还有文字。", [_image_bytes(28, 28), _image_bytes(300, 19)]),
    ],
)
def test_fast_labels_match_old_route_for_image_placeholder_layouts(
    processor: Any,
    content: str,
    images: list[Any],
) -> None:
    sample = _sample(
        [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "图像布局回答。"},
            {"role": "user", "content": "再追问一轮。"},
            {"role": "assistant", "content": "第二轮回答。"},
        ],
        images=images,
    )
    _assert_fast_matches_old(sample, processor=processor)


@pytest.mark.parametrize("thinking_mode", [True, False, "non-empty", None])
def test_fast_labels_match_old_route_for_thinking_modes(
    processor: Any,
    thinking_mode: bool | None | str,
) -> None:
    sample = _sample(
        [
            {"role": "user", "content": "解释空 think 和普通答案。"},
            {"role": "assistant", "content": "<think>\n\n</think>\n\n最终答案。"},
            {"role": "user", "content": "再回答一次。"},
            {"role": "assistant", "content": "没有显式 think 的答案。"},
        ]
    )
    _assert_fast_matches_old(sample, processor=processor, thinking_mode=thinking_mode)


def test_fast_label_route_uses_image_processor_once(processor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        pytest.skip("processor has no image_processor")

    call_count = 0
    original_call = type(image_processor).__call__

    def counted_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(type(image_processor), "__call__", counted_call)
    sample = _sample(
        [
            {"role": "user", "content": "只处理一次图像：<image>"},
            {"role": "assistant", "content": "确认。"},
        ],
        images=[_image_bytes(64, 64)],
    )
    processed = process_sample(sample, processor=processor, max_length=DEFAULT_MAX_LENGTH)

    assert call_count == 1
    assert "pixel_values" in processed
    assert "image_grid_thw" in processed


@pytest.mark.parametrize(
    "sample",
    [
        _sample(
            [{"role": "user", "content": "少图 <image>"}, {"role": "assistant", "content": "不会到这里。"}],
            images=[],
        ),
        _sample(
            [{"role": "user", "content": "多图但无占位。"}, {"role": "assistant", "content": "不会到这里。"}],
            images=[_image_bytes(16, 16)],
        ),
        _sample(
            [{"role": "critic", "content": "bad"}, {"role": "assistant", "content": "不会到这里。"}],
            images=[],
        ),
        _sample(
            [{"role": "user", "content": {"not": "string"}}, {"role": "assistant", "content": "不会到这里。"}],
            images=[],
        ),
    ],
)
def test_invalid_samples_still_return_skipped_sentinel(processor: Any, sample: dict[str, Any]) -> None:
    processed = process_sample(sample, processor=processor, max_length=DEFAULT_MAX_LENGTH)
    assert int(processed["input_ids"].numel()) == 0
    assert int(processed["attention_mask"].numel()) == 0
    assert int(processed["labels"].numel()) == 0


def test_fast_label_mismatch_returns_skipped_sentinel(processor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    original = dataset_module._build_expanded_label_messages

    def broken_build_expanded_label_messages(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        messages = original(*args, **kwargs)
        messages[0]["content"].append({"type": "text", "text": "force mismatch"})
        return messages

    monkeypatch.setattr(dataset_module, "_build_expanded_label_messages", broken_build_expanded_label_messages)
    sample = _sample(
        [
            {"role": "user", "content": "正常样本。"},
            {"role": "assistant", "content": "正常回答。"},
        ]
    )

    processed = process_sample(sample, processor=processor, max_length=DEFAULT_MAX_LENGTH)
    assert int(processed["input_ids"].numel()) == 0
    assert int(processed["attention_mask"].numel()) == 0
    assert int(processed["labels"].numel()) == 0


@pytest.mark.skipif(
    os.environ.get("OPENBEE_RUN_SLOW_LABEL_BENCH") != "1",
    reason="set OPENBEE_RUN_SLOW_LABEL_BENCH=1 to run large-image timing coverage",
)
def test_slow_large_image_fast_label_benchmark(processor: Any) -> None:
    for image_count in (1, 2, 3):
        sample = _sample(
            [
                {"role": "user", "content": " ".join(["大图 <image>"] * image_count)},
                {"role": "assistant", "content": "第一轮大图回答。"},
                {"role": "user", "content": "继续比较。"},
                {"role": "assistant", "content": "第二轮回答。"},
            ],
            images=[_image_bytes(3000, 2200, (20 + idx, 80, 140)) for idx in range(image_count)],
        )

        old_start = time.perf_counter()
        old = _old_reference_process_sample(sample, processor=processor, max_length=32768)
        old_elapsed = time.perf_counter() - old_start

        fast_start = time.perf_counter()
        fast = process_sample(sample, processor=processor, max_length=32768)
        fast_elapsed = time.perf_counter() - fast_start

        assert torch.equal(fast["labels"], old["labels"])
        speedup = old_elapsed / fast_elapsed if fast_elapsed else float("inf")
        print(
            f"openbee_fast_label_bench images={image_count} "
            f"old={old_elapsed:.3f}s new={fast_elapsed:.3f}s speedup={speedup:.2f}x"
        )
