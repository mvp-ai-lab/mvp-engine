"""Configuration exports for the Magic Transformer recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import MagicTransformerConfig

__all__ = ["MagicTransformerConfig"]


def __getattr__(name: str):
    """Lazily resolve Magic Transformer config exports."""
    if name != "MagicTransformerConfig":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .schema import MagicTransformerConfig

    return MagicTransformerConfig
