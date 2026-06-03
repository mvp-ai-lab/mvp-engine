"""Lazy exports for logging backends."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import Backend
    from .file import FileBackend
    from .terminal import TerminalBackend
    from .wandb import WandbBackend

__all__ = ["Backend", "FileBackend", "TerminalBackend", "WandbBackend"]

_EXPORT_MODULES = {
    "Backend": ".backend",
    "FileBackend": ".file",
    "TerminalBackend": ".terminal",
    "WandbBackend": ".wandb",
}


def __getattr__(name: str):
    """Lazily resolve logging backend exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
