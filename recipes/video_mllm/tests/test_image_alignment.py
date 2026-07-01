"""Image-text alignment (single-image) tests for the video MLLM recipe.

Validates that OpenBee-style image rows are normalized into one OneVision visual
slot and encoded as a single frame on the same patch-sequence path as video.
"""

import torch
from PIL import Image

from mvp_engine.kit import MLLMTokenizationHandler
from recipes.video_mllm.dataset.image_encoding import process_image_as_frame, read_image
from recipes.video_mllm.dataset.media import (
    OneVisionImageHandler,
    VideoMLLMMediaHandler,
)
from recipes.video_mllm.dataset.schema import VideoChatSchemaHandler
from recipes.video_mllm.dataset.video_encoding import DenseVideoConfig


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [200 if ch == "V" else ord(ch) for ch in text]}


class _FakeImageProcessor:
    def __call__(self, images, return_tensors="pt", do_resize=True, do_center_crop=True):
        del return_tensors, do_resize, do_center_crop
        import torchvision.transforms.v2.functional as tvF

        pixel_values = torch.stack([tvF.pil_to_tensor(image).float() for image in images], dim=0)
        return {"pixel_values": pixel_values}


class _FakeProcessor:
    video_token = "V"
    video_token_id = 200
    tokenizer = _FakeTokenizer()
    onevision_image_processor = _FakeImageProcessor()
    onevision_patch_size = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        has_assistant = any(message["role"] == "assistant" for message in messages)
        text = "P" + self.video_token
        if has_assistant:
            text += "ANS"
        return text


def _image_row() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>describe"},
            {"role": "assistant", "content": "ANS"},
        ],
        "images": ["demo.jpg"],
        "img_size": [[16, 16]],
    }


def test_schema_binds_image_field_as_single_visual_slot():
    segments, slots, _ = VideoChatSchemaHandler(_FakeProcessor()).normalize(_image_row())

    assert len(slots) == 1
    assert slots[0].media_type == "video"
    assert slots[0].field == "images"
    assert slots[0].index == 0
    assert sum(1 for segment in segments if segment.type == "video") == 1


def test_process_image_as_frame_emits_single_frame_layout():
    config = DenseVideoConfig(num_frames=1, frame_size=4, patch_size=2)
    result = process_image_as_frame(
        Image.new("RGB", (12, 9), color=128),
        processor=_FakeProcessor(),
        config=config,
    )

    grid = config.grid_size  # 2
    assert result.patch_values.shape == (grid * grid, 3, 2, 2)
    assert result.token_positions.shape == (grid * grid, 3)
    assert torch.all(result.token_positions[:, 0] == 0)  # single frame -> t == 0
    assert result.frame_grid_thw.tolist() == [[1, grid, grid]]
    assert result.visual_token_count == grid * grid


def test_image_handler_expands_video_pads_and_masks_labels():
    processor = _FakeProcessor()
    image_config = DenseVideoConfig(num_frames=1, frame_size=4, patch_size=2)
    handler = OneVisionImageHandler(image_config=image_config)
    media_handler = VideoMLLMMediaHandler(processor=processor, video_handler=handler)

    segments, slots, _ = VideoChatSchemaHandler(processor).normalize(_image_row())
    rendered_segments = media_handler.render(segments, slots)
    input_ids, labels, attention_mask = MLLMTokenizationHandler(
        processor=processor,
        max_seq_len=64,
    ).tokenize(rendered_segments)
    model_media = handler.load(slots, [Image.new("RGB", (16, 16), color=64)], processor=processor)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    grid_tokens = image_config.grid_size**2  # 4

    assert int((input_ids == 200).sum().item()) == grid_tokens
    assert torch.all(labels[input_ids == 200] == -100)
    assert int((labels != -100).sum().item()) == len("ANS")
    assert model_media["pixel_values_videos"].shape == (grid_tokens, 3, 2, 2)
    assert model_media["video_grid_thw"].tolist() == [[1, grid_tokens, 1]]
    assert model_media["video_token_positions"].shape == (grid_tokens, 3)
    assert model_media["video_token_counts"].tolist() == [grid_tokens]
    assert torch.equal(torch.tensor(attention_mask), torch.ones_like(input_ids))


def test_read_image_accepts_pil_and_bytes():
    import io

    pil = Image.new("RGB", (8, 8), color=200)
    assert read_image(pil).size == (8, 8)

    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    assert read_image(buffer.getvalue()).size == (8, 8)
