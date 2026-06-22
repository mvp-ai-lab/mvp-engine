"""General-purpose utility kits."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .step_counting import StepCountingKit, StepCountResult

__all__ = [
    "StepCountResult",
    "StepCountingKit",
]

_EXPORT_MODULES = {
    "StepCountResult": ".step_counting",
    "StepCountingKit": ".step_counting",
}


def __getattr__(name: str):
    """Lazily resolve utility kit exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
