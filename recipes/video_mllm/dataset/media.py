"""Video media handlers for the video MLLM DataKit pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mvp_engine.kit import MLLMMediaHandler, MLLMMediaSlot, MLLMMediaTypeHandler
from mvp_engine.kit.mllm.data.media import RenderedMedia, empty_model_sample
from mvp_engine.utils.log import simple_info

from .codec import CodecPatchConfig, process_video_with_codec
from .image_encoding import process_image_as_frame
from .schema import DEFAULT_VIDEO_TOKEN
from .video_encoding import (
    DenseVideoConfig,
    KeyframeLowresVideoConfig,
    VideoEncodingResult,
    process_video_with_dense_frames,
    process_video_with_keyframe_lowres,
)


class VideoMLLMMediaHandler(MLLMMediaHandler):
    """Media registry for the recipe's single-video OneVision path."""

    def __init__(self, processor: Any, video_handler: MLLMMediaTypeHandler) -> None:
        """Register the video handler under the DataKit ``video`` media type."""
        super().__init__(processor=processor, handlers={"video": video_handler})


class OneVisionVideoHandler(MLLMMediaTypeHandler):
    """Render, decode, pack, and collate OneVision-backed video tensors."""

    media_type = "video"
    media_noun = "video"  # used in drop-warning messages; overridden by the image handler
    OUTPUT_TENSOR_KEYS = (
        "pixel_values_videos",
        "video_grid_thw",
        "video_token_positions",
        "video_frame_grid_thw",
        "video_merge_sizes",
    )

    def __init__(
        self,
        *,
        strategy: str,
        dense_config: DenseVideoConfig | None = None,
        codec_config: CodecPatchConfig | None = None,
        keyframe_config: KeyframeLowresVideoConfig | None = None,
        video_root: str | None = None,
    ) -> None:
        """Store strategy-specific video encoding options."""
        self.strategy = strategy
        self.dense_config = dense_config
        self.codec_config = codec_config
        self.keyframe_config = keyframe_config
        self.video_root = video_root
        self._validate_strategy()

    def render(self, slot: MLLMMediaSlot, *, processor: Any, tokenizer: Any) -> RenderedMedia:
        """Render one video slot as the repeated model video token span."""
        del tokenizer
        return RenderedMedia(
            media_id=slot.media_id,
            media_type=self.media_type,
            text=self.default_token(processor) * self.video_token_count,
        )

    def default_token(self, processor: Any) -> str:
        """Return the processor video token."""
        video_token = getattr(processor, "video_token", DEFAULT_VIDEO_TOKEN)
        if not isinstance(video_token, str) or not video_token:
            raise ValueError("processor must expose a valid video token.")
        return video_token

    def placeholder_aliases(self, processor: Any) -> tuple[str, ...]:
        """Return raw placeholder strings accepted by the recipe schema."""
        return ("<video>", self.default_token(processor))

    @property
    def video_token_count(self) -> int:
        """Return the fixed language-side video token count for the active strategy."""
        if self.strategy == "uniform":
            assert self.dense_config is not None
            return int(self.dense_config.num_frames * self.dense_config.grid_size * self.dense_config.grid_size)
        if self.strategy == "codec_patch":
            assert self.codec_config is not None
            return int(self.codec_config.k_keep)
        if self.strategy == "keyframe_lowres":
            assert self.keyframe_config is not None
            count = 0
            for frame_index in range(int(self.keyframe_config.num_frames)):
                if self.keyframe_config.is_keyframe(frame_index):
                    count += int(self.keyframe_config.full_grid_size**2)
                else:
                    count += int(self.keyframe_config.lowres_grid_size**2)
            return count
        raise ValueError(f"unsupported video encoding strategy: {self.strategy!r}")

    def load(self, slots: list[MLLMMediaSlot], values: list[Any], *, processor: Any) -> dict[str, Any]:
        """Decode video references into OneVision model-input tensors."""
        if not values:
            return {}
        try:
            outputs = [self._encode_video(value, processor=processor) for value in values]
            return self._merge_video_inputs([output.to_model_inputs() for output in outputs])
        except Exception as exc:
            simple_info(f"video_mllm: dropping sample with unreadable {self.media_noun} media: {exc}", level="warning")
            return empty_model_sample()

    def merge_pack(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge per-source video tensors into one packed sample."""
        return self._merge_video_inputs(samples)

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate packed video tensors into one model batch."""
        collated = self._merge_video_inputs(batch)
        self._validate_video_layout(collated)
        return collated

    def _encode_video(self, video: Any, *, processor: Any) -> VideoEncodingResult:
        """Run the active video encoding strategy on one raw media value."""
        video_path = self._resolve_video_path(video)
        if self.strategy == "uniform":
            assert self.dense_config is not None
            return process_video_with_dense_frames(video_path, processor=processor, config=self.dense_config)
        if self.strategy == "codec_patch":
            assert self.codec_config is not None
            output = process_video_with_codec(video_path, processor=processor, config=self.codec_config)
            if output.visual_token_count != int(self.codec_config.k_keep):
                raise ValueError(
                    f"codec output contains {output.visual_token_count} tokens but k_keep={self.codec_config.k_keep}."
                )
            return output
        if self.strategy == "keyframe_lowres":
            assert self.keyframe_config is not None
            return process_video_with_keyframe_lowres(video_path, processor=processor, config=self.keyframe_config)
        raise ValueError(f"unsupported video encoding strategy: {self.strategy!r}")

    def _resolve_video_path(self, video: Any) -> str:
        """Resolve one raw video value to a path string."""
        if isinstance(video, dict):
            video = video.get("path", video.get("video", video.get("value")))
        if not isinstance(video, str) or not video:
            raise ValueError(f"contains an invalid video value: {type(video).__name__}.")

        path = Path(video).expanduser()
        if self.video_root is not None and not path.is_absolute():
            path = Path(self.video_root).expanduser() / path
        return str(path)

    def _merge_video_inputs(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge video tensors from source or packed samples in placeholder order."""
        merged: dict[str, Any] = {}
        for key in self.OUTPUT_TENSOR_KEYS:
            tensors = [sample[key] for sample in samples if isinstance(sample.get(key), torch.Tensor)]
            if tensors:
                merged[key] = torch.cat(tensors, dim=0)

        token_counts = []
        for sample in samples:
            if isinstance(sample.get("video_token_counts"), torch.Tensor):
                token_counts.append(sample["video_token_counts"].reshape(-1).to(dtype=torch.long))
            elif isinstance(sample.get("visual_token_count"), torch.Tensor):
                token_counts.append(sample["visual_token_count"].reshape(1).to(dtype=torch.long))
        if token_counts:
            merged["video_token_counts"] = torch.cat(token_counts, dim=0)

        frame_counts = []
        for sample in samples:
            if isinstance(sample.get("video_frame_counts"), torch.Tensor):
                frame_counts.append(sample["video_frame_counts"].reshape(-1).to(dtype=torch.long))
            elif isinstance(sample.get("video_frame_grid_thw"), torch.Tensor):
                frame_counts.append(torch.tensor([int(sample["video_frame_grid_thw"].shape[0])], dtype=torch.long))
        if frame_counts:
            merged["video_frame_counts"] = torch.cat(frame_counts, dim=0)

        return merged

    def _validate_video_layout(self, model_inputs: dict[str, Any]) -> None:
        """Validate concatenated OneVision tensor/layout consistency."""
        pixel_values = model_inputs.get("pixel_values_videos")
        if not isinstance(pixel_values, torch.Tensor):
            return
        for key in ("video_grid_thw", "video_token_positions", "video_token_counts"):
            if key not in model_inputs:
                raise ValueError(f"video batch is missing required `{key}` layout metadata.")
        visual_token_count = int(pixel_values.shape[0])
        if int(model_inputs["video_token_positions"].shape[0]) != visual_token_count:
            raise ValueError("video_token_positions length must match pixel_values_videos rows.")
        if int(model_inputs["video_token_counts"].sum().item()) != visual_token_count:
            raise ValueError("video_token_counts must sum to pixel_values_videos rows.")
        if int(model_inputs["video_grid_thw"].prod(dim=-1).sum().item()) != visual_token_count:
            raise ValueError("video_grid_thw must imply the concatenated visual token count.")

    def _validate_strategy(self) -> None:
        """Validate the active strategy has exactly the config it needs."""
        if self.strategy == "uniform":
            if self.dense_config is None:
                raise ValueError("dense_config is required for uniform video encoding.")
            self.dense_config.validate()
            return
        if self.strategy == "codec_patch":
            if self.codec_config is None:
                raise ValueError("codec_config is required for codec_patch video encoding.")
            return
        if self.strategy == "keyframe_lowres":
            if self.keyframe_config is None:
                raise ValueError("keyframe_config is required for keyframe_lowres video encoding.")
            self.keyframe_config.validate()
            return
        raise ValueError(f"unsupported video encoding strategy: {self.strategy!r}")


class OneVisionImageHandler(OneVisionVideoHandler):
    """Encode a single still image as one OneVision frame for image-text alignment.

    Reuses the validated video patch-sequence path: the image is rendered with the
    video token and emitted as ``pixel_values_videos`` so the model routes it through
    ``get_video_features`` with no model changes. One image == one ``uniform`` frame.
    """

    media_noun = "image"

    def __init__(self, *, image_config: DenseVideoConfig, image_root: str | None = None) -> None:
        """Store the single-frame image geometry."""
        super().__init__(strategy="uniform", dense_config=image_config, video_root=image_root)

    def placeholder_aliases(self, processor: Any) -> tuple[str, ...]:
        """Accept the image placeholder in addition to the video token."""
        return ("<image>", "<video>", self.default_token(processor))

    def _encode_video(self, video: Any, *, processor: Any) -> VideoEncodingResult:
        """Encode one raw image value as a single OneVision frame."""
        assert self.dense_config is not None
        return process_image_as_frame(
            video,
            processor=processor,
            config=self.dense_config,
            image_root=self.video_root,
        )


class OneVisionVisualHandler(OneVisionVideoHandler):
    """Mixed image+video handler: dispatch each row by its source field.

    OpenBee image rows (``image``/``images`` field) are encoded as one OneVision
    frame; OV2 video rows (``video``/``videos``/``images_source``) go through the
    configured video strategy. Both render the video token and emit
    ``pixel_values_videos`` so the model uses one path (``get_video_features``).
    Used for Stage-2 mixed mid-training (image+video in one dataset).
    """

    media_noun = "visual"
    IMAGE_FIELDS = ("image", "images")

    def __init__(
        self,
        *,
        strategy: str,
        image_config: DenseVideoConfig,
        dense_config: DenseVideoConfig | None = None,
        codec_config: CodecPatchConfig | None = None,
        keyframe_config: KeyframeLowresVideoConfig | None = None,
        video_root: str | None = None,
    ) -> None:
        """Store the video strategy config plus the single-frame image config."""
        super().__init__(
            strategy=strategy,
            dense_config=dense_config,
            codec_config=codec_config,
            keyframe_config=keyframe_config,
            video_root=video_root,
        )
        image_config.validate()
        self.image_config = image_config

    @staticmethod
    def _is_image_slot(slot: MLLMMediaSlot) -> bool:
        """Return whether a slot is an OpenBee image (vs an OV2 video) by source field."""
        return slot.field in OneVisionVisualHandler.IMAGE_FIELDS

    @property
    def _image_token_count(self) -> int:
        return int(self.image_config.grid_size * self.image_config.grid_size)

    def render(self, slot: MLLMMediaSlot, *, processor: Any, tokenizer: Any) -> RenderedMedia:
        """Render one slot's video-token span sized for image (1 frame) or video (strategy)."""
        del tokenizer
        count = self._image_token_count if self._is_image_slot(slot) else self.video_token_count
        return RenderedMedia(
            media_id=slot.media_id, media_type=self.media_type, text=self.default_token(processor) * count
        )

    def load(self, slots: list[MLLMMediaSlot], values: list[Any], *, processor: Any) -> dict[str, Any]:
        """Encode each slot by its kind (image → 1 frame, video → strategy) and merge."""
        if not values:
            return {}
        try:
            outputs: list[VideoEncodingResult] = []
            for slot, value in zip(slots, values, strict=True):
                if self._is_image_slot(slot):
                    outputs.append(
                        process_image_as_frame(
                            value, processor=processor, config=self.image_config, image_root=self.video_root
                        )
                    )
                else:
                    outputs.append(self._encode_video(value, processor=processor))
            return self._merge_video_inputs([output.to_model_inputs() for output in outputs])
        except Exception as exc:
            simple_info(f"video_mllm: dropping sample with unreadable {self.media_noun} media: {exc}", level="warning")
            return empty_model_sample()


def build_video_media_handler(config: Any, *, processor: Any) -> VideoMLLMMediaHandler:
    """Build the recipe video media handler from config."""
    patch_size = int(getattr(processor, "onevision_patch_size", 14))
    data = config.data
    strategy = str(data.video_encoding_strategy)
    dense_config = codec_config = keyframe_config = None
    if strategy == "uniform":
        dense_config = DenseVideoConfig(
            num_frames=int(data.num_frames),
            frame_size=int(data.video_frame_size),
            patch_size=patch_size,
        )
    elif strategy == "codec_patch":
        codec_config = CodecPatchConfig(
            num_frames=int(data.codec_num_frames),
            packed_frames=int(data.codec_packed_frames),
            frame_size=int(data.codec_frame_size),
            patch_size=int(data.codec_patch_size),
            k_keep=int(data.codec_k_keep),
            cv_reader_required=bool(data.cv_reader_required),
        )
    elif strategy == "keyframe_lowres":
        keyframe_config = KeyframeLowresVideoConfig(
            num_frames=int(data.num_frames),
            full_frame_size=int(data.video_frame_size),
            lowres_frame_size=int(data.keyframe_lowres_frame_size),
            patch_size=patch_size,
            keyframe_interval=int(data.keyframe_interval),
        )

    return VideoMLLMMediaHandler(
        processor=processor,
        video_handler=OneVisionVideoHandler(
            strategy=strategy,
            dense_config=dense_config,
            codec_config=codec_config,
            keyframe_config=keyframe_config,
            video_root=data.video_root,
        ),
    )


def build_image_media_handler(config: Any, *, processor: Any) -> VideoMLLMMediaHandler:
    """Build the recipe image media handler (single-frame OneVision encoding)."""
    patch_size = int(getattr(processor, "onevision_patch_size", 14))
    data = config.data
    image_config = DenseVideoConfig(num_frames=1, frame_size=int(data.video_frame_size), patch_size=patch_size)
    return VideoMLLMMediaHandler(
        processor=processor,
        video_handler=OneVisionImageHandler(image_config=image_config, image_root=data.video_root),
    )


def build_mixed_media_handler(config: Any, *, processor: Any) -> VideoMLLMMediaHandler:
    """Build the Stage-2 mixed handler (per-row image+video dispatch).

    Video rows use ``video_encoding_strategy`` at ``video_frame_size``; image rows
    use ``image_frame_size`` as a single frame. Both stream through one OneVision
    video path.
    """
    patch_size = int(getattr(processor, "onevision_patch_size", 14))
    data = config.data
    strategy = str(data.video_encoding_strategy)
    dense_config = codec_config = keyframe_config = None
    if strategy == "uniform":
        dense_config = DenseVideoConfig(
            num_frames=int(data.num_frames), frame_size=int(data.video_frame_size), patch_size=patch_size
        )
    elif strategy == "codec_patch":
        codec_config = CodecPatchConfig(
            num_frames=int(data.codec_num_frames),
            packed_frames=int(data.codec_packed_frames),
            frame_size=int(data.codec_frame_size),
            patch_size=int(data.codec_patch_size),
            k_keep=int(data.codec_k_keep),
            cv_reader_required=bool(data.cv_reader_required),
        )
    elif strategy == "keyframe_lowres":
        keyframe_config = KeyframeLowresVideoConfig(
            num_frames=int(data.num_frames),
            full_frame_size=int(data.video_frame_size),
            lowres_frame_size=int(data.keyframe_lowres_frame_size),
            patch_size=patch_size,
            keyframe_interval=int(data.keyframe_interval),
        )
    image_config = DenseVideoConfig(num_frames=1, frame_size=int(data.image_frame_size), patch_size=patch_size)

    return VideoMLLMMediaHandler(
        processor=processor,
        video_handler=OneVisionVisualHandler(
            strategy=strategy,
            image_config=image_config,
            dense_config=dense_config,
            codec_config=codec_config,
            keyframe_config=keyframe_config,
            video_root=data.video_root,
        ),
    )
