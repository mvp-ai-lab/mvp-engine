"""Model helpers for the Magic Transformer recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .builder import build_magic_transformer_model
    from .magic_transformer import MagicTransformer, TransformerConfig

__all__ = [
    "MagicTransformer",
    "TransformerConfig",
    "build_magic_transformer_model",
]

_EXPORT_MODULES = {
    "MagicTransformer": ".magic_transformer",
    "TransformerConfig": ".magic_transformer",
    "build_magic_transformer_model": ".builder",
}


def __getattr__(name: str):
    """Lazily resolve Magic Transformer model exports."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
