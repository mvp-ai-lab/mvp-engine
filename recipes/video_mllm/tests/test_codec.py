"""Codec patchification tests for the video MLLM recipe."""

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from mvp_dataset.cache.fingerprint import callable_fingerprint
from omegaconf import OmegaConf

from recipes.video_mllm.configs.schema import VideoMLLMConfig
from recipes.video_mllm.dataset import codec as codec_module
from recipes.video_mllm.dataset.codec import (
    CodecPatchConfig,
    _load_cv_reader_residual_arrays,
    _load_residuals,
    _residual_arrays_to_tensor,
    indices_to_patch_positions,
    mask_by_residual_topk,
    pack_video_patches,
    process_video_with_codec,
)
from recipes.video_mllm.dataset.collator import VideoMLLMCollator
from recipes.video_mllm.dataset.video_encoding import (
    KeyframeLowresVideoConfig,
    VideoEncodingResult,
    process_video_with_keyframe_lowres,
)
from recipes.video_mllm.model.onevision import OneVisionVisualTower

CODEC_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "codec.yaml"


def _load_codec_config() -> VideoMLLMConfig:
    container = OmegaConf.to_container(OmegaConf.load(CODEC_CONFIG), resolve=True)
    return VideoMLLMConfig.model_validate(container)


def test_codec_config_validates_against_schema():
    config = _load_codec_config()

    assert config.data.video_encoding_strategy == "codec_patch"
    assert config.data.cv_reader_required is False
    grid = config.data.codec_frame_size // config.data.codec_patch_size
    assert config.data.codec_k_keep == config.data.codec_packed_frames * grid * grid
    assert config.model.vision_encoder_name_or_path
    assert config.model.freeze_vision_encoder is True


def test_codec_schema_rejects_mismatched_k_keep():
    container = OmegaConf.to_container(OmegaConf.load(CODEC_CONFIG), resolve=True)
    container["data"]["codec_k_keep"] = container["data"]["codec_k_keep"] - 1

    with pytest.raises(ValueError):
        VideoMLLMConfig.model_validate(container)


def test_schema_accepts_keyframe_lowres_strategy():
    container = OmegaConf.to_container(OmegaConf.load(CODEC_CONFIG), resolve=True)
    container["data"]["video_encoding_strategy"] = "keyframe_lowres"
    container["data"]["keyframe_interval"] = 2
    container["data"]["keyframe_lowres_frame_size"] = 112

    config = VideoMLLMConfig.model_validate(container)

    assert config.data.uses_keyframe_lowres is True
    assert config.data.keyframe_interval == 2
    assert config.data.keyframe_lowres_frame_size == 112


def test_schema_rejects_keyframe_lowres_larger_than_full_resolution():
    container = OmegaConf.to_container(OmegaConf.load(CODEC_CONFIG), resolve=True)
    container["data"]["video_encoding_strategy"] = "keyframe_lowres"
    container["data"]["video_frame_size"] = 112
    container["data"]["keyframe_lowres_frame_size"] = 224

    with pytest.raises(ValueError, match="keyframe_lowres_frame_size"):
        VideoMLLMConfig.model_validate(container)


def test_build_uniform_sample_expands_dense_video_pads_and_masks_labels(monkeypatch):
    from recipes.video_mllm.dataset import preprocess as preprocess_module
    from recipes.video_mllm.dataset.video_encoding import DenseVideoConfig

    config = DenseVideoConfig(num_frames=2, frame_size=4, patch_size=2)

    fake_outputs = VideoEncodingResult(
        patch_values=torch.zeros(8, 3, 2, 2),
        token_positions=torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        ),
        frame_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long),
        merge_sizes=torch.ones(2, dtype=torch.long),
    )
    monkeypatch.setattr(preprocess_module, "process_video_with_dense_frames", lambda *a, **k: fake_outputs)

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            ids = [200 if ch == "V" else ord(ch) for ch in text]
            return {"input_ids": ids}

    class FakeProcessor:
        video_token = "V"
        video_token_id = 200
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            has_assistant = any(m["role"] == "assistant" for m in messages)
            text = "P" + FakeProcessor.video_token
            if has_assistant and not add_generation_prompt:
                text += "ANS"
            return text

    sample = {
        "messages": [
            {"role": "user", "content": "<video>describe"},
            {"role": "assistant", "content": "ANS"},
        ],
        "video": "demo.mp4",
    }

    out = preprocess_module._build_uniform_sample(sample, processor=FakeProcessor(), max_length=64, dense_config=config)

    input_ids = out["input_ids"]
    labels = out["labels"]
    assert int((input_ids == 200).sum().item()) == 8
    assert torch.all(labels[input_ids == 200] == -100)
    assert int((labels != -100).sum().item()) == len("ANS")
    assert out["video_grid_thw"].tolist() == [[1, 8, 1]]
    assert out["pixel_values_videos"].shape == (8, 3, 2, 2)
    assert out["video_token_positions"].shape == (8, 3)
    assert int(out["visual_token_count"].item()) == 8
    assert torch.equal(out["attention_mask"], torch.ones_like(input_ids))


def test_build_codec_sample_expands_video_pads_and_masks_labels(monkeypatch):
    from recipes.video_mllm.dataset import preprocess as preprocess_module

    config = CodecPatchConfig(num_frames=2, packed_frames=1, frame_size=4, patch_size=2, k_keep=4)

    fake_outputs = VideoEncodingResult(
        patch_values=torch.zeros(4, 3, 2, 2),
        token_positions=torch.zeros(4, 3, dtype=torch.long),
        frame_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long),
        merge_sizes=torch.ones(2, dtype=torch.long),
    )
    monkeypatch.setattr(preprocess_module, "process_video_with_codec", lambda *a, **k: fake_outputs)

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            # Tokenize char-by-char, mapping the (single-char stand-in) video pad to its id.
            ids = [200 if ch == "V" else ord(ch) for ch in text]
            return {"input_ids": ids}

    class FakeProcessor:
        # Single-char video token keeps the char tokenizer trivial; one pad per video block.
        video_token = "V"
        video_token_id = 200
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            # One leading prompt char + the single video pad, then the assistant answer when present.
            has_assistant = any(m["role"] == "assistant" for m in messages)
            text = "P" + FakeProcessor.video_token
            if has_assistant and not add_generation_prompt:
                text += "ANS"
            return text

    sample = {
        "messages": [
            {"role": "user", "content": "<video>describe"},
            {"role": "assistant", "content": "ANS"},
        ],
        "video": "demo.mp4",
    }

    out = preprocess_module._build_codec_sample(sample, processor=FakeProcessor(), max_length=64, codec_config=config)

    input_ids = out["input_ids"]
    labels = out["labels"]
    assert int((input_ids == 200).sum().item()) == config.k_keep
    # All video pads masked out of the loss.
    assert torch.all(labels[input_ids == 200] == -100)
    # Prompt prefix (leading "P" + 4 pads) masked; "ANS" supervised.
    assert int((labels != -100).sum().item()) == len("ANS")
    assert out["video_token_positions"].shape == (config.k_keep, 3)
    assert out["video_grid_thw"].tolist() == [[1, config.k_keep, 1]]
    assert out["pixel_values_videos"].shape == (config.k_keep, 3, 2, 2)
    assert int(out["visual_token_count"].item()) == config.k_keep
    assert torch.equal(out["attention_mask"], torch.ones_like(input_ids))


def test_build_keyframe_lowres_sample_expands_actual_visual_tokens(monkeypatch):
    from recipes.video_mllm.dataset import preprocess as preprocess_module

    config = KeyframeLowresVideoConfig(
        num_frames=3,
        full_frame_size=4,
        lowres_frame_size=2,
        patch_size=1,
        keyframe_interval=2,
    )
    fake_outputs = VideoEncodingResult(
        patch_values=torch.zeros(36, 3, 1, 1),
        token_positions=torch.zeros(36, 3),
        frame_grid_thw=torch.tensor([[1, 4, 4], [1, 2, 2], [1, 4, 4]], dtype=torch.long),
        merge_sizes=torch.ones(3, dtype=torch.long),
    )
    monkeypatch.setattr(preprocess_module, "process_video_with_keyframe_lowres", lambda *a, **k: fake_outputs)

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            ids = [200 if ch == "V" else ord(ch) for ch in text]
            return {"input_ids": ids}

    class FakeProcessor:
        video_token = "V"
        video_token_id = 200
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            has_assistant = any(m["role"] == "assistant" for m in messages)
            text = "P" + FakeProcessor.video_token
            if has_assistant and not add_generation_prompt:
                text += "ANS"
            return text

    sample = {
        "messages": [
            {"role": "user", "content": "<video>describe"},
            {"role": "assistant", "content": "ANS"},
        ],
        "video": "demo.mp4",
    }

    out = preprocess_module._build_keyframe_lowres_sample(
        sample,
        processor=FakeProcessor(),
        max_length=128,
        keyframe_config=config,
    )

    input_ids = out["input_ids"]
    labels = out["labels"]
    assert int((input_ids == 200).sum().item()) == 36
    assert torch.all(labels[input_ids == 200] == -100)
    assert int((labels != -100).sum().item()) == len("ANS")
    assert out["video_grid_thw"].tolist() == [[1, 36, 1]]
    assert out["video_frame_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 2], [1, 4, 4]]


def test_collator_concats_visual_token_layout_and_counts():
    first_video = VideoEncodingResult(
        patch_values=torch.zeros(2, 3, 2, 2),
        token_positions=torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long),
        frame_grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.long),
        merge_sizes=torch.ones(1, dtype=torch.long),
    ).to_model_inputs()
    second_video = VideoEncodingResult(
        patch_values=torch.ones(3, 3, 2, 2),
        token_positions=torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.long),
        frame_grid_thw=torch.tensor([[1, 2, 1], [1, 1, 1]], dtype=torch.long),
        merge_sizes=torch.ones(2, dtype=torch.long),
    ).to_model_inputs()
    batch = [
        {
            "input_ids": torch.tensor([1, 200, 200, 2]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([-100, -100, -100, 2]),
            **first_video,
        },
        {
            "input_ids": torch.tensor([1, 200, 200, 200, 3]),
            "attention_mask": torch.ones(5, dtype=torch.long),
            "labels": torch.tensor([-100, -100, -100, -100, 3]),
            **second_video,
        },
    ]

    out = VideoMLLMCollator(pad_token_id=0)(batch)

    assert out["pixel_values_videos"].shape == (5, 3, 2, 2)
    assert out["video_token_positions"].tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]
    assert out["video_token_counts"].tolist() == [2, 3]
    assert out["video_grid_thw"].tolist() == [[1, 2, 1], [1, 3, 1]]
    assert out["video_frame_grid_thw"].tolist() == [[1, 1, 2], [1, 2, 1], [1, 1, 1]]
    assert out["video_frame_counts"].tolist() == [1, 2]
    assert out["video_merge_sizes"].tolist() == [1, 1, 1]
    assert out["total_tokens"] == 9
    assert out["effective_tokens"] == 2


def test_onevision_patch_sequence_uses_per_sample_token_positions():
    class FakeRope:
        def __init__(self):
            self.position_shapes = []

        def forward_from_positions(self, positions):
            # Real OneVision video_rope takes [batch, seq, 3] and returns [batch, seq, half].
            self.position_shapes.append(tuple(positions.shape))
            return torch.zeros(positions.shape[0], positions.shape[1], 2, device=positions.device)

    class FakeEncoderBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_shapes = []

        def forward(self, hidden_states, *, attention_mask, rotary_pos_emb, **kwargs):
            self.rotary_shapes.append(tuple(rotary_pos_emb.shape))
            return SimpleNamespace(last_hidden_state=hidden_states)

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = SimpleNamespace(patch_embedding=nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False))
            self.video_rope = FakeRope()
            self.layernorm_pre = nn.Identity()
            self.encoder = FakeEncoderBlock()
            self.layernorm_post = None

    tower = OneVisionVisualTower.__new__(OneVisionVisualTower)
    nn.Module.__init__(tower)
    tower.encoder = FakeEncoder()
    tower.projection = nn.Identity()

    outputs = tower(
        torch.zeros(5, 3, 2, 2),
        token_positions=torch.tensor(
            [[0, 0, 0], [0, 0, 1], [2, 1, 0], [2, 1, 1], [3, 0, 0]],
            dtype=torch.long,
        ),
        token_counts=torch.tensor([2, 3], dtype=torch.long),
    )

    assert outputs.pooler_output.shape == (5, 4)
    assert tower.encoder.video_rope.position_shapes == [(1, 2, 3), (1, 3, 3)]
    assert tower.encoder.encoder.rotary_shapes == [(1, 2, 4), (1, 3, 4)]


def test_keyframe_lowres_video_processor_outputs_dense_variable_resolution_tokens(monkeypatch):
    from recipes.video_mllm.dataset import video_encoding as video_encoding_module

    config = KeyframeLowresVideoConfig(
        num_frames=3,
        full_frame_size=4,
        lowres_frame_size=2,
        patch_size=1,
        keyframe_interval=2,
    )
    frames = [
        torch.zeros(3, 4, 4, dtype=torch.uint8),
        torch.zeros(3, 2, 2, dtype=torch.uint8),
        torch.zeros(3, 4, 4, dtype=torch.uint8),
    ]
    monkeypatch.setattr(
        video_encoding_module,
        "load_keyframe_lowres_video_frames",
        lambda *_args, **_kwargs: (frames, [True, False, True]),
    )

    class FakeImageProcessor:
        def __call__(self, *, images, return_tensors, do_resize, do_center_crop):
            assert len(images) == 1
            assert return_tensors == "pt"
            assert do_resize is False
            assert do_center_crop is False
            height, width = images[0].height, images[0].width
            return {"pixel_values": torch.zeros(1, 3, height, width)}

    class FakeProcessor:
        onevision_image_processor = FakeImageProcessor()

    outputs = process_video_with_keyframe_lowres("demo.mp4", processor=FakeProcessor(), config=config)

    assert outputs.patch_values.shape == (36, 3, 1, 1)
    assert outputs.model_video_grid_thw.tolist() == [[1, 36, 1]]
    assert outputs.frame_grid_thw.tolist() == [[1, 4, 4], [1, 2, 2], [1, 4, 4]]
    assert outputs.merge_sizes.tolist() == [1, 1, 1]
    assert outputs.token_positions.shape == (36, 3)
    assert torch.allclose(
        outputs.token_positions[16:20, 1:], torch.tensor([[0.0, 0.0], [0.0, 3.0], [3.0, 0.0], [3.0, 3.0]])
    )


def test_residual_topk_is_sorted_and_in_bounds():
    residuals = torch.zeros(1, 1, 2, 4, 4)
    residuals[0, 0, 0, :2, :2] = 10
    residuals[0, 0, 1, 2:, 2:] = 20

    indices = mask_by_residual_topk(residuals, k_keep=2, patch_size=2)

    assert indices.shape == (1, 2)
    assert torch.equal(indices, torch.tensor([[0, 7]]))
    assert torch.all(indices >= 0)
    assert torch.all(indices < 8)


def test_indices_to_patch_positions_matches_flattened_layout():
    indices = torch.tensor([0, 3, 4, 7])
    positions = indices_to_patch_positions(indices, grid_h=2, grid_w=2)

    assert torch.equal(
        positions,
        torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
            ]
        ),
    )


def test_pack_video_patches_shape_and_order():
    config = CodecPatchConfig(num_frames=2, packed_frames=1, frame_size=4, patch_size=2, k_keep=4)
    video = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    visible_indices = torch.tensor([0, 1, 4, 7])

    packed = pack_video_patches(video, visible_indices, config)

    assert packed.shape == (1, 3, 4, 4)
    assert torch.equal(packed[:, :, :2, :2], video[:1, :, :2, :2])
    assert torch.equal(packed[:, :, :2, 2:], video[:1, :, :2, 2:])
    assert torch.equal(packed[:, :, 2:, :2], video[1:2, :, :2, :2])
    assert torch.equal(packed[:, :, 2:, 2:], video[1:2, :, 2:, 2:])


def test_patch_positions_length_matches_visual_tokens():
    config = CodecPatchConfig(num_frames=64, packed_frames=8, frame_size=224, patch_size=14, k_keep=2048)
    visible_indices = torch.arange(config.k_keep)

    positions = indices_to_patch_positions(visible_indices, grid_h=config.grid_size, grid_w=config.grid_size)

    assert positions.shape == (config.k_keep, 3)


def test_codec_config_is_cache_fingerprintable():
    def fn(*, codec_config):
        return codec_config.k_keep

    config = CodecPatchConfig()
    fingerprint = callable_fingerprint(partial(fn, codec_config=config))

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64


def test_codec_video_processor_keeps_packed_frame_size(monkeypatch, tmp_path):
    config = CodecPatchConfig(num_frames=2, packed_frames=1, frame_size=4, patch_size=2, k_keep=4)
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")

    frames = torch.arange(2 * 3 * 4 * 4, dtype=torch.uint8).reshape(2, 3, 4, 4)
    residuals = torch.zeros(1, 1, 2, 4, 4)
    residuals[0, 0, 0] = 10

    monkeypatch.setattr(codec_module, "_load_video_frames", lambda *_args, **_kwargs: frames)
    monkeypatch.setattr(codec_module, "_load_residuals", lambda *_args, **_kwargs: residuals)

    class FakeImageProcessor:
        def __call__(self, *, images, return_tensors, do_resize, do_center_crop):
            assert return_tensors == "pt"
            assert do_resize is False
            assert do_center_crop is False
            height, width = images[0].height, images[0].width
            return {"pixel_values": torch.zeros(len(images), 3, height, width)}

    class FakeProcessor:
        onevision_image_processor = FakeImageProcessor()

    outputs = process_video_with_codec(video_path, processor=FakeProcessor(), config=config)

    assert outputs.patch_values.shape == (4, 3, 2, 2)
    assert outputs.model_video_grid_thw.tolist() == [[1, 4, 1]]
    assert outputs.token_positions.shape == (4, 3)


def test_cv_reader_callback_requests_selected_frame_ids():
    requested = {}

    class FakeCvApi:
        @staticmethod
        def read_video_cb(path, callback, without_residual, max_frames, frame_ids):
            requested["path"] = path
            requested["without_residual"] = without_residual
            requested["max_frames"] = max_frames
            requested["frame_ids"] = frame_ids
            for frame_id in frame_ids:
                callback(
                    {"frame_idx": frame_id, "residual_y": torch.full((4, 4), 128 + frame_id, dtype=torch.uint8).numpy()}
                )

    residuals = _load_cv_reader_residual_arrays("demo.mp4", frame_indices=[0, 2, 2], cv_api=FakeCvApi())

    assert requested == {
        "path": "demo.mp4",
        "without_residual": 0,
        "max_frames": 3,
        "frame_ids": [0, 2, 2],
    }
    assert len(residuals) == 3
    assert int(residuals[1][0, 0]) == 130


def test_cv_reader_full_decode_fallback_uses_read_video():
    class FakeCvApi:
        @staticmethod
        def read_video(path, without_residual, max_frames):
            assert path == "demo.mp4"
            assert without_residual == 0
            assert max_frames == 3
            return [
                {"residual_y": torch.full((4, 4), 128, dtype=torch.uint8).numpy()},
                {"residual_y": torch.full((4, 4), 129, dtype=torch.uint8).numpy()},
                {"residual_y": torch.full((4, 4), 130, dtype=torch.uint8).numpy()},
            ]

    residuals = _load_cv_reader_residual_arrays("demo.mp4", frame_indices=[0, 2], cv_api=FakeCvApi())

    assert len(residuals) == 2
    assert int(residuals[1][0, 0]) == 130


def test_residual_arrays_to_tensor_shape_and_centering():
    config = CodecPatchConfig(num_frames=1, packed_frames=1, frame_size=4, patch_size=2, k_keep=4)
    residuals = _residual_arrays_to_tensor([torch.full((4, 4), 130, dtype=torch.uint8).numpy()], config)

    assert residuals.shape == (1, 1, 1, 4, 4)
    assert torch.all(residuals == 2)


def test_required_codec_uses_cv_reader_for_h264_and_hevc(monkeypatch):
    calls = []

    def fake_cv_reader(video_path, *, frame_indices, cv_api=None):
        calls.append((video_path, frame_indices))
        return [torch.full((4, 4), 130, dtype=torch.uint8).numpy()]

    monkeypatch.setattr(codec_module, "_probe_video_frame_count", lambda _path: 1)
    monkeypatch.setattr(codec_module, "_load_cv_reader_residual_arrays", fake_cv_reader)
    config = CodecPatchConfig(
        num_frames=1, packed_frames=1, frame_size=4, patch_size=2, k_keep=4, cv_reader_required=True
    )

    monkeypatch.setattr(codec_module, "_probe_video_codec", lambda _path: "h264")
    h264 = _load_residuals("h264.mp4", config)
    monkeypatch.setattr(codec_module, "_probe_video_codec", lambda _path: "hevc")
    hevc = _load_residuals("hevc.mp4", config)

    assert h264.shape == (1, 1, 1, 4, 4)
    assert hevc.shape == (1, 1, 1, 4, 4)
    assert calls == [("h264.mp4", [0]), ("hevc.mp4", [0])]


def test_required_codec_fails_when_cv_reader_is_missing(monkeypatch):
    monkeypatch.setattr(codec_module, "_probe_video_codec", lambda _path: "h264")
    monkeypatch.setattr(codec_module, "_probe_video_frame_count", lambda _path: 1)

    def missing_cv_reader(*_args, **_kwargs):
        raise RuntimeError("cv_reader missing")

    monkeypatch.setattr(codec_module, "_load_cv_reader_residual_arrays", missing_cv_reader)
    config = CodecPatchConfig(
        num_frames=1, packed_frames=1, frame_size=4, patch_size=2, k_keep=4, cv_reader_required=True
    )

    try:
        _load_residuals("demo.mp4", config)
    except RuntimeError as exc:
        assert "cv_reader" in str(exc)
    else:
        raise AssertionError("missing cv_reader should fail when required")


def test_optional_non_hevc_video_uses_frame_difference_fallback(monkeypatch):
    monkeypatch.setattr(codec_module, "_probe_video_codec", lambda _path: "vp9")
    fallback = torch.zeros(1, 1, 1, 4, 4)
    monkeypatch.setattr(codec_module, "_load_frame_difference_residuals", lambda *_args, **_kwargs: fallback)
    config = CodecPatchConfig(
        num_frames=1, packed_frames=1, frame_size=4, patch_size=2, k_keep=4, cv_reader_required=False
    )

    residuals = _load_residuals("demo.mp4", config)

    assert residuals is fallback


def test_optional_cv_reader_failure_uses_frame_difference_fallback(monkeypatch):
    monkeypatch.setattr(codec_module, "_probe_video_codec", lambda _path: "h264")
    monkeypatch.setattr(codec_module, "_probe_video_frame_count", lambda _path: 1)
    monkeypatch.setattr(
        codec_module,
        "_load_cv_reader_residual_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    fallback = torch.zeros(1, 1, 1, 4, 4)
    monkeypatch.setattr(codec_module, "_load_frame_difference_residuals", lambda *_args, **_kwargs: fallback)
    config = CodecPatchConfig(
        num_frames=1, packed_frames=1, frame_size=4, patch_size=2, k_keep=4, cv_reader_required=False
    )

    residuals = _load_residuals("demo.mp4", config)

    assert residuals is fallback
