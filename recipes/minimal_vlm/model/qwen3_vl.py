"""Qwen3-VL model helpers for the minimal VLM recipe."""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForImageTextToText, AutoProcessor


def build_qwen3_vl_processor(model_config: Any):
    """Load the processor for the configured Qwen3-VL checkpoint."""
    processor = AutoProcessor.from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=bool(getattr(model_config, "trust_remote_code", True)),
    )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    return processor


def freeze_visual_parameters(model, *, freeze_visual: bool = True) -> int:
    """Freeze the Qwen3-VL visual stack in-place and return the frozen parameter count."""
    if not freeze_visual:
        return 0

    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        if name.startswith("model.visual."):
            parameter.requires_grad = False
            frozen_parameters += parameter.numel()
    return frozen_parameters


def build_qwen3_vl_model(model_config: Any):
    """Load the Qwen3-VL model and apply the recipe freeze policy."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=bool(getattr(model_config, "trust_remote_code", True)),
        torch_dtype="auto",
    )
    freeze_visual_parameters(model, freeze_visual=bool(getattr(model_config, "freeze_visual", True)))
    return model
