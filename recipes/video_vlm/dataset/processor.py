"""Processor helpers for the Video VLM recipe."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoImageProcessor, AutoProcessor


class ProcessorFingerprint:
    """Pickle-safe callable that returns a stable processor fingerprint."""

    def __init__(self, value: str):
        """Store the precomputed fingerprint string."""
        self.value = value

    def __call__(self) -> str:
        """Return the stored fingerprint."""
        return self.value


def build_qwen3_vl_processor(model_config: Any):
    """Load the Qwen3-VL processor and normalize tokenizer padding.

    Args:
        model_config: Recipe model config with the pretrained model reference.

    Returns:
        The initialized Hugging Face processor for Qwen3-VL.
    """
    processor = AutoProcessor.from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=True,
    )
    vision_encoder_name = getattr(model_config, "vision_encoder_name_or_path", None)
    if vision_encoder_name:
        vision_config = AutoConfig.from_pretrained(vision_encoder_name, trust_remote_code=True)
        processor.onevision_image_processor = AutoImageProcessor.from_pretrained(
            vision_encoder_name,
            trust_remote_code=True,
        )
        processor.onevision_patch_size = int(getattr(vision_config, "patch_size", 14))
        processor.onevision_image_size = int(getattr(vision_config, "image_size", 448))
    image_processor = getattr(processor, "image_processor", None)
    image_max_pixels = getattr(model_config, "image_max_pixels", None)
    if image_processor is not None and image_max_pixels is not None:
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            size["longest_edge"] = int(image_max_pixels)
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = int(image_max_pixels)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    processor.__fingerprint__ = ProcessorFingerprint(_processor_fingerprint(processor))
    return processor


def _processor_fingerprint(processor: Any) -> str:
    """Return a stable cache fingerprint for a HF processor."""
    candidates = [
        getattr(processor, "name_or_path", None),
        getattr(getattr(processor, "tokenizer", None), "name_or_path", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            base = candidate
            break
    else:
        base = f"{type(processor).__module__}.{type(processor).__qualname__}"

    image_processor = getattr(processor, "image_processor", None)
    image_size = getattr(image_processor, "size", None)
    onevision_image_processor = getattr(processor, "onevision_image_processor", None)
    if onevision_image_processor is not None:
        image_size = getattr(onevision_image_processor, "size", image_size)
    if isinstance(image_size, dict):
        shortest_edge = image_size.get("shortest_edge")
        longest_edge = image_size.get("longest_edge")
        return f"{base}|image_size={shortest_edge}x{longest_edge}"
    return base
