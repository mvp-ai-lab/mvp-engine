"""Processor helpers for the Qwen + OneVision video data path."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoImageProcessor


def attach_onevision_processor(processor: Any, model_config: Any) -> Any:
    """Attach the OneVision image processor required by video preprocessing."""
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
    return processor
