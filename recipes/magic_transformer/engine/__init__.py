"""Engine exports for the Magic Transformer recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .magic_transformer_engine import MagicTransformerEngine

__all__ = ["MagicTransformerEngine"]


def __getattr__(name: str):
    """Lazily resolve Magic Transformer engine exports."""
    if name != "MagicTransformerEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .magic_transformer_engine import MagicTransformerEngine

    return MagicTransformerEngine
