"""MLLM-specific kits."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import (
        MLLMCollator,
        MLLMDataKit,
        MLLMMediaKit,
        MLLMSampleKit,
        ModelInputs,
        PackingOptions,
    )
    from .model import MLLMModelKit

__all__ = [
    "MLLMCollator",
    "MLLMDataKit",
    "MLLMMediaKit",
    "MLLMModelKit",
    "MLLMSampleKit",
    "ModelInputs",
    "PackingOptions",
]

_EXPORT_MODULES = {
    "MLLMCollator": ".data",
    "MLLMDataKit": ".data",
    "MLLMMediaKit": ".data",
    "MLLMModelKit": ".model",
    "MLLMSampleKit": ".data",
    "ModelInputs": ".data",
    "PackingOptions": ".data",
}


def __getattr__(name: str):
    """Lazily resolve MLLM kit exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
