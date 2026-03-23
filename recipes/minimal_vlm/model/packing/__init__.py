"""Packed-attention helpers for the minimal VLM recipe."""

from .fa2_patch import segmented_flash_attention_patches
from .prepare import prepare_packed_model_inputs

__all__ = [
    "prepare_packed_model_inputs",
    "segmented_flash_attention_patches",
]
