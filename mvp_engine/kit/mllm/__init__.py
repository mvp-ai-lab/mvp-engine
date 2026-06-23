"""MLLM-specific kits."""

# ruff: noqa: F401

from typing import TYPE_CHECKING

from mvp_engine.kit._lazy import resolve_lazy_export

if TYPE_CHECKING:
    from .data import MLLMDataKit
    from .model import MLLMModelKit
    from .utils import MLLMStepEstimationKit

_KIT_MODULES = {
    "MLLMDataKit": ".data",
    "MLLMModelKit": ".model",
    "MLLMStepEstimationKit": ".utils",
}

__all__ = list(_KIT_MODULES)


def __getattr__(name: str):
    return resolve_lazy_export(globals(), _KIT_MODULES, name)
