"""Engine registry and lazy engine exports."""

from typing import TYPE_CHECKING

from mvp_engine.utils.registry import Registry

if TYPE_CHECKING:
    from .engine import Engine, TrainStepContext

ENGINE_REGISTRY = Registry()


__all__ = ["Engine", "ENGINE_REGISTRY", "TrainStepContext"]


def __getattr__(name: str):
    """Lazily resolve engine exports without importing the engine module eagerly."""
    if name in {"Engine", "TrainStepContext"}:
        from .engine import Engine, TrainStepContext

        exports = {
            "Engine": Engine,
            "TrainStepContext": TrainStepContext,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
