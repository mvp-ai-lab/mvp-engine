"""MLLM utility kits."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .step_estimation import Confidence, MLLMStepEstimationKit, StepEstimateResult

__all__ = [
    "Confidence",
    "MLLMStepEstimationKit",
    "StepEstimateResult",
]

_EXPORT_MODULES = {
    "Confidence": ".step_estimation",
    "MLLMStepEstimationKit": ".step_estimation",
    "StepEstimateResult": ".step_estimation",
}


def __getattr__(name: str):
    """Lazily resolve MLLM utility kit exports from their implementation modules."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
