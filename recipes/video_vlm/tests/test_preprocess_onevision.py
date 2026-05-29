"""Preprocess tests for OneVision image token accounting."""

from types import SimpleNamespace

from recipes.video_vlm.dataset import preprocess as preprocess_module
from recipes.video_vlm.dataset.codec import CodecPatchConfig
from recipes.video_vlm.dataset.preprocess import (
    convert_images_to_pixel_values,
    process_sample,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<|vision_start|>":
            return {"input_ids": [101]}
        if text == "<|vision_end|>":
            return {"input_ids": [102]}
        if text == "<|image_pad|>":
            return {"input_ids": [103]}
        if text == "<|video_pad|>":
            return {"input_ids": [104]}
        if "<|image_pad|>" in text:
            token_count = text.count("<|image_pad|>") + len(text.replace("<|image_pad|>", " ").split())
            return {"input_ids": list(range(token_count))}
        return {"input_ids": list(range(max(1, len(text.split()))))}


class FakeProcessor(SimpleNamespace):
    image_token = "<|image_pad|>"
    video_token = "<|video_pad|>"
    tokenizer = FakeTokenizer()
    onevision_patch_size = 14
    onevision_image_size = 448
    onevision_image_processor = object()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        text = []
        for message in messages:
            for block in message["content"]:
                if block["type"] == "text":
                    text.append(block["text"])
                elif block["type"] == "image":
                    text.append(self.image_token)
        return " ".join(text)


def test_onevision_image_placeholder_expands_to_fixed_grid():
    sample = {
        "messages": [
            {"role": "user", "content": "<image>\nWhat is shown?"},
            {"role": "assistant", "content": "A demo image."},
        ],
        "images": ["demo.png"],
        "image_size": [[900, 600]],
    }

    processed = process_sample(
        sample,
        processor=FakeProcessor(),
        max_length=4096,
        thinking_mode=None,
    )

    assert processed["adjusted_image_size"] == [[448, 448]]
    assert processed["input_ids"].numel() >= 1024


def test_required_codec_failure_is_not_silently_dropped(monkeypatch):
    def fail_codec(*_args, **_kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(preprocess_module, "process_video_with_codec", fail_codec)
    sample = {"videos": ["demo.mp4"], "input_ids": []}
    config = CodecPatchConfig(cv_reader_required=True)

    try:
        convert_images_to_pixel_values(sample, processor=FakeProcessor(), codec_config=config)
    except RuntimeError as exc:
        assert "decoder failed" in str(exc)
    else:
        raise AssertionError("required codec failures must be raised")


def test_optional_codec_failure_builds_empty_sample(monkeypatch):
    def fail_codec(*_args, **_kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(preprocess_module, "process_video_with_codec", fail_codec)
    sample = {"videos": ["demo.mp4"], "input_ids": []}
    config = CodecPatchConfig(cv_reader_required=False)

    processed = convert_images_to_pixel_values(sample, processor=FakeProcessor(), codec_config=config)

    assert processed["input_ids"].numel() == 0
