"""Processor helpers for the minimal VLM recipe."""

from __future__ import annotations

from typing import Any

from transformers import AutoProcessor


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

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    return processor
