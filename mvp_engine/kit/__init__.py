"""Reusable training kits."""

# ruff: noqa: F401

from typing import TYPE_CHECKING

from mvp_engine.kit._lazy import resolve_lazy_export

if TYPE_CHECKING:
    from .llm.data import LLMDataKit
    from .llm.model import LLMModelKit
    from .llm.utils import LLMStepEstimationKit
    from .loss.loss import LossKit
    from .loss.token_loss import TokenNormedLossKit
    from .mllm.data import MLLMDataKit
    from .mllm.model import MLLMModelKit
    from .mllm.utils import MLLMStepEstimationKit
    from .model import LigerKernelKit
    from .optim import OptimKit
    from .perf import MFUKit
    from .util import StepCountingKit

_KIT_MODULES = {
    "LLMDataKit": ".llm.data",
    "LLMModelKit": ".llm.model",
    "LLMStepEstimationKit": ".llm.utils",
    "LossKit": ".loss.loss",
    "TokenNormedLossKit": ".loss.token_loss",
    "MLLMDataKit": ".mllm.data",
    "MLLMModelKit": ".mllm.model",
    "MLLMStepEstimationKit": ".mllm.utils",
    "LigerKernelKit": ".model",
    "MFUKit": ".perf",
    "OptimKit": ".optim",
    "StepCountingKit": ".util",
}

__all__ = list(_KIT_MODULES)


def __getattr__(name: str):
    return resolve_lazy_export(globals(), _KIT_MODULES, name)
