"""Processor helpers for the PanguVL recipe."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from transformers import AutoProcessor


class ProcessorFingerprint:
    """Pickle-safe callable that returns a stable processor fingerprint."""

    def __init__(self, value: str):
        self.value = value

    def __call__(self) -> str:
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
    processor_module = importlib.import_module(processor.__class__.__module__)
    if not hasattr(processor_module, "np"):
        processor_module.np = np

    image_processor = getattr(processor, "image_processor", None)
    image_max_pixels = getattr(model_config, "image_max_pixels", None)
    if image_processor is not None:
        size = getattr(image_processor, "size", None)
        if size is not None and not isinstance(size, Mapping):
            image_processor.size = dict(vars(size))
            size = image_processor.size
        if image_max_pixels is not None and isinstance(size, Mapping):
            size["longest_edge"] = int(image_max_pixels)
        if isinstance(size, Mapping):
            image_processor.min_pixels = size.get("shortest_edge")
            image_processor.max_pixels = size.get("longest_edge")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        pad_token_id = getattr(model_config, "pad_token_id", 2)
        if tokenizer.pad_token_id is None:
            pad_token_id = int(pad_token_id)
            pad_token = tokenizer.convert_ids_to_tokens(pad_token_id)
            if pad_token is None or pad_token == tokenizer.unk_token:
                raise ValueError(f"Cannot set pad_token: token id {pad_token_id} is not a known tokenizer token.")
            tokenizer.pad_token = pad_token
            tokenizer.pad_token_id = pad_token_id

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
    if isinstance(image_size, Mapping):
        shortest_edge = image_size.get("shortest_edge")
        longest_edge = image_size.get("longest_edge")
        return f"{base}|image_size={shortest_edge}x{longest_edge}"
    return base
