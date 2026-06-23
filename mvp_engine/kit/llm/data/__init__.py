"""LLM data-kit entrypoint."""

# ruff: noqa: F401

from typing import TYPE_CHECKING

from mvp_engine.kit._lazy import resolve_lazy_export

if TYPE_CHECKING:
    from .data import LLMDataKit

_KIT_MODULES = {
    "LLMDataKit": ".data",
}

__all__ = list(_KIT_MODULES)


def __getattr__(name: str):
    return resolve_lazy_export(globals(), _KIT_MODULES, name)
