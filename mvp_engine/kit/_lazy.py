"""Internal helpers for lazy kit entrypoint exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def resolve_lazy_export(module_globals: dict[str, Any], export_modules: dict[str, str], name: str) -> Any:
    """Resolve and cache one lazy export from a kit entrypoint module."""
    if name not in export_modules:
        module_name = module_globals.get("__name__", "<unknown>")
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    value = getattr(import_module(export_modules[name], module_globals["__name__"]), name)
    module_globals[name] = value
    return value
