"""Processor helpers for the video MLLM recipe."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoImageProcessor, AutoProcessor


def build_qwen3_vl_processor(model_config: Any):
    """Load the Qwen3-VL processor plus OneVision image processor.

    The Qwen3-VL tokenizer/chat template is kept, while all video pixels are
    normalized by the OneVision image processor attached as
    ``processor.onevision_image_processor`` / ``onevision_patch_size`` /
    ``onevision_image_size``.

    Args:
        model_config: Recipe model config with the pretrained model and OneVision encoder references.

    Returns:
        The initialized Hugging Face processor for Qwen3-VL.
    """
    processor = AutoProcessor.from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=True,
    )

    vision_encoder_name = getattr(model_config, "vision_encoder_name_or_path", None)
    if not vision_encoder_name:
        raise ValueError("video MLLM preprocessing requires `model.vision_encoder_name_or_path`.")
    vision_config = AutoConfig.from_pretrained(vision_encoder_name, trust_remote_code=True)
    processor.onevision_image_processor = AutoImageProcessor.from_pretrained(
        vision_encoder_name,
        trust_remote_code=True,
    )
    processor.onevision_patch_size = int(getattr(vision_config, "patch_size", 14))
    processor.onevision_image_size = int(getattr(vision_config, "image_size", 448))

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    return processor
