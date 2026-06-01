"""Processor helpers for the video MLLM recipe."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoImageProcessor, AutoProcessor


def build_qwen3_vl_processor(
    model_config: Any,
    *,
    video_encoding_strategy: str = "uniform",
    vision_encoder_backend: str = "qwen3_vl",
):
    """Load the Qwen3-VL processor (image + video) and normalize tokenizer padding.

    When the OneVision backend is active, the OneVision image processor and its patch/image geometry are attached as
    ``processor.onevision_image_processor`` / ``onevision_patch_size`` /
    ``onevision_image_size`` so supported video strategies can pixel-normalize
    frames with the encoder's own statistics. The native Qwen3-VL path is untouched.

    Args:
        model_config: Recipe model config with the pretrained model reference and optional visual encoder reference.
        video_encoding_strategy: Recipe-local data encoding strategy name.
        vision_encoder_backend: Visual tower backend name.

    Returns:
        The initialized Hugging Face processor for Qwen3-VL.
    """
    processor = AutoProcessor.from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=True,
    )

    if vision_encoder_backend == "onevision":
        vision_encoder_name = getattr(model_config, "vision_encoder_name_or_path", None)
        if not vision_encoder_name:
            raise ValueError("OneVision preprocessing requires `model.vision_encoder_name_or_path`.")
        if video_encoding_strategy != "codec_patch":
            raise ValueError(
                "OneVision preprocessing currently supports only `data.video_encoding_strategy=codec_patch`."
            )
        vision_config = AutoConfig.from_pretrained(vision_encoder_name, trust_remote_code=True)
        processor.onevision_image_processor = AutoImageProcessor.from_pretrained(
            vision_encoder_name,
            trust_remote_code=True,
        )
        processor.onevision_patch_size = int(getattr(vision_config, "patch_size", 14))
        processor.onevision_image_size = int(getattr(vision_config, "image_size", 448))
    elif vision_encoder_backend != "qwen3_vl":
        raise ValueError(f"unsupported vision encoder backend: {vision_encoder_backend!r}")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    return processor
