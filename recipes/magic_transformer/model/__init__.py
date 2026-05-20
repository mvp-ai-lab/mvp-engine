"""Model helpers for the Magic Transformer recipe."""

from .builder import build_magic_transformer_model
from .magic_transformer import MagicTransformer, TransformerConfig

__all__ = [
    "MagicTransformer",
    "TransformerConfig",
    "build_magic_transformer_model",
]
