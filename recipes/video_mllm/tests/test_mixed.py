"""Per-row image+video dispatch tests for Stage-2 mixed mid-training.

Validates that one ``OneVisionVisualHandler`` routes each row by its source field
(OpenBee ``images`` → 1 OneVision frame; OV2 ``video``/``images_source`` → strategy
frames), both emitting ``pixel_values_videos`` for the shared get_video_features path.
"""

import torch
from PIL import Image

from mvp_engine.kit import MLLMMediaSlot
from recipes.video_mllm.dataset import media as media_module
from recipes.video_mllm.dataset.media import OneVisionVisualHandler
from recipes.video_mllm.dataset.video_encoding import (
    DenseVideoConfig,
    VideoEncodingResult,
)


class _FakeImageProcessor:
    def __call__(self, images, return_tensors="pt", do_resize=True, do_center_crop=True):
        del return_tensors, do_resize, do_center_crop
        import torchvision.transforms.v2.functional as tvF

        return {"pixel_values": torch.stack([tvF.pil_to_tensor(im).float() for im in images], dim=0)}


class _FakeProcessor:
    video_token = "V"
    onevision_image_processor = _FakeImageProcessor()
    onevision_patch_size = 2


def _fake_video(n_frames: int, grid: int) -> VideoEncodingResult:
    n = n_frames * grid * grid
    return VideoEncodingResult(
        patch_values=torch.zeros(n, 3, 2, 2),
        token_positions=torch.zeros(n, 3, dtype=torch.long),
        frame_grid_thw=torch.tensor([[1, grid, grid]] * n_frames, dtype=torch.long),
        merge_sizes=torch.ones(n_frames, dtype=torch.long),
    )


def _handler() -> OneVisionVisualHandler:
    return OneVisionVisualHandler(
        strategy="uniform",
        image_config=DenseVideoConfig(num_frames=1, frame_size=4, patch_size=2),  # grid 2 -> 4 tokens
        dense_config=DenseVideoConfig(num_frames=3, frame_size=4, patch_size=2),  # 3 frames -> 12 tokens
    )


def test_field_dispatch_image_vs_video():
    h = _handler()
    assert h._is_image_slot(MLLMMediaSlot(media_id="video:0", media_type="video", field="images", index=0))
    assert h._is_image_slot(MLLMMediaSlot(media_id="video:0", media_type="video", field="image"))
    # OV2's images_source is a VIDEO path despite the name -> must NOT be image
    assert not h._is_image_slot(MLLMMediaSlot(media_id="video:0", media_type="video", field="images_source"))
    assert not h._is_image_slot(MLLMMediaSlot(media_id="video:0", media_type="video", field="video"))


def test_render_token_counts_differ_by_kind():
    h, proc = _handler(), _FakeProcessor()
    img = h.render(
        MLLMMediaSlot(media_id="video:0", media_type="video", field="images", index=0), processor=proc, tokenizer=None
    )
    vid = h.render(MLLMMediaSlot(media_id="video:0", media_type="video", field="video"), processor=proc, tokenizer=None)
    assert img.text == "V" * 4  # image grid^2
    assert vid.text == "V" * 12  # 3 frames * grid^2


def test_load_dispatches_and_emits_video_tensors(monkeypatch):
    h, proc = _handler(), _FakeProcessor()
    monkeypatch.setattr(media_module, "process_video_with_dense_frames", lambda *a, **k: _fake_video(3, 2))

    img_out = h.load(
        [MLLMMediaSlot(media_id="video:0", media_type="video", field="images", index=0)],
        [Image.new("RGB", (8, 8), color=120)],
        processor=proc,
    )
    assert img_out["pixel_values_videos"].shape[0] == 4
    assert img_out["video_token_counts"].tolist() == [4]

    vid_out = h.load(
        [MLLMMediaSlot(media_id="video:0", media_type="video", field="video")],
        ["demo.mp4"],
        processor=proc,
    )
    assert vid_out["pixel_values_videos"].shape[0] == 12
    assert vid_out["video_token_counts"].tolist() == [12]
