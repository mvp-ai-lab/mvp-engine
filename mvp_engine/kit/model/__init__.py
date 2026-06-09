"""Model transformation helpers for reusable training kits."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .liger import LigerKernelKit, LigerKernelReport, LigerPatch

__all__ = [
    "LigerKernelKit",
    "LigerKernelReport",
    "LigerPatch",
]

_EXPORT_MODULES = {
    "LigerKernelKit": ".liger",
    "LigerKernelReport": ".liger",
    "LigerPatch": ".liger",
}


def __getattr__(name: str):
    """Lazily resolve model-kit exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
