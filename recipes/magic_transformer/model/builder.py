"""Builder utilities for the Magic Transformer recipe."""

from __future__ import annotations

from ..configs.schema import MagicTransformerModelConfig
from .magic_transformer import MagicTransformer, TransformerConfig


def build_magic_transformer_model(config: MagicTransformerModelConfig) -> MagicTransformer:
    """Convert the recipe config model into the recipe-local model dataclass."""
    model_config = TransformerConfig(**config.model_dump())
    return MagicTransformer(model_config)
