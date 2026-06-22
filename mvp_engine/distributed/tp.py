from collections.abc import Callable
from typing import Any

import torch.nn as nn

from mvp_engine.utils.log import logger

try:
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        ParallelStyle,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )
except Exception as exc:  # pragma: no cover - runtime-dependent
    raise ImportError("Please install pytorch >= 2.3.0 for tensor parallel support.") from exc

TPModulePostprocessor = Callable[[nn.Module, Any], None]


def _scale_grad(scale: float) -> Callable[[Any], Any]:
    def hook(grad):
        if grad is None:
            return None
        return grad * scale

    return hook


def _scale_accumulated_grad_delta(scale: float) -> Callable[[nn.Parameter], None]:
    state: dict[str, Any] = {}

    def hook(parameter: nn.Parameter) -> None:
        grad = parameter.grad
        if grad is None:
            state.clear()
            return

        grad_id = id(grad)
        previous = state.get("previous")
        previous_grad_id = state.get("grad_id")
        if previous is None or previous_grad_id != grad_id:
            grad.mul_(scale)
        else:
            delta = grad - previous
            grad.copy_(previous + delta * scale)

        state["previous"] = grad.detach().clone()
        state["grad_id"] = grad_id

    return hook


def attach_sequence_parallel_grad_scale(model: nn.Module) -> nn.Module:
    """Attach TP builtin sequence-parallel gradient scaling to the final model."""
    scale = getattr(model, "_tp_sequence_parallel_grad_scale", 1.0)
    names = getattr(model, "_tp_sequence_parallel_grad_scale_names", ())
    if scale == 1.0 or not names:
        return model

    names = set(names)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name in names:
            if hasattr(parameter, "register_post_accumulate_grad_hook"):
                parameter.register_post_accumulate_grad_hook(_scale_accumulated_grad_delta(float(scale)))
            else:
                parameter.register_hook(_scale_grad(float(scale)))
    return model


def _build_tp_plan(
    plan_cfg: object,
    sequence_parallel: bool = False,
    sequence_dim: int = 1,
) -> dict[str, ParallelStyle]:
    if not isinstance(plan_cfg, dict):
        raise TypeError(f"Unsupported TP plan config type: {type(plan_cfg)}")

    expected_modes = ("col", "row", "sequence") if sequence_parallel else ("col", "row")
    plan: dict[str, ParallelStyle] = {}
    for child_name, mode in plan_cfg.items():
        if not isinstance(mode, str):
            raise TypeError(f"TP mode for {child_name} must be str, got {type(mode)}.")

        mode = mode.lower()
        if mode not in expected_modes:
            raise ValueError(f"Unknown TP mode '{mode}' for {child_name}. Expected one of {expected_modes}.")

        if mode == "sequence":
            plan[child_name] = SequenceParallel(sequence_dim=sequence_dim, use_local_output=True)
        elif mode == "col" and sequence_parallel:
            plan[child_name] = ColwiseParallel(input_layouts=Shard(sequence_dim))
        elif mode == "row" and sequence_parallel:
            plan[child_name] = RowwiseParallel(output_layouts=Shard(sequence_dim))
        elif mode == "col":
            plan[child_name] = ColwiseParallel()
        else:
            plan[child_name] = RowwiseParallel()
    return plan


def _resolve_tp_module_config(
    model: nn.Module,
    attr_name: str = "TP_MODULE_CONFIG",
    required: bool = True,
) -> dict[str, object]:
    """Load a model-defined tensor-parallel module config from a class attribute."""
    cls = model.__class__
    if not hasattr(cls, attr_name):
        if not required:
            return {}
        raise AttributeError(
            f"{cls.__name__} does not define class attribute '{attr_name}'. "
            "Please provide TP module plan via model class attribute."
        )

    module_config = getattr(cls, attr_name)
    if module_config is None:
        if not required:
            return {}
        raise ValueError(f"{cls.__name__}.{attr_name} is None.")
    if not isinstance(module_config, dict):
        raise TypeError(f"{cls.__name__}.{attr_name} must be dict, got {type(module_config)}.")

    return module_config


def _resolve_sequence_parallel_sequence_dim(
    model: nn.Module,
    attr_name: str = "SEQUENCE_PARALLEL_SEQUENCE_DIM",
) -> int:
    """Load the sequence dimension used by sequence parallel layouts."""
    cls = model.__class__
    sequence_dim = getattr(cls, attr_name, 1)
    if not isinstance(sequence_dim, int):
        raise TypeError(f"{cls.__name__}.{attr_name} must be int, got {type(sequence_dim)}.")
    return sequence_dim


def _resolve_sequence_parallel_module_sequence_dims(
    model: nn.Module,
    attr_name: str = "SEQUENCE_PARALLEL_MODULE_SEQUENCE_DIMS",
) -> dict[str, int]:
    cls = model.__class__
    module_sequence_dims = getattr(cls, attr_name, {})
    if module_sequence_dims is None:
        return {}
    if not isinstance(module_sequence_dims, dict):
        raise TypeError(f"{cls.__name__}.{attr_name} must be dict, got {type(module_sequence_dims)}.")
    for module_name, sequence_dim in module_sequence_dims.items():
        if not isinstance(module_name, str):
            raise TypeError(f"{cls.__name__}.{attr_name} keys must be str, got {type(module_name)}.")
        if not isinstance(sequence_dim, int):
            raise TypeError(f"{cls.__name__}.{attr_name}[{module_name!r}] must be int, got {type(sequence_dim)}.")
    return module_sequence_dims


def _merge_tp_module_configs(
    tp_config: dict[str, object],
    sequence_parallel_config: dict[str, object],
) -> dict[str, object]:
    """Merge TP and sequence-parallel module configs keyed by runtime class name."""
    merged = dict(tp_config)
    for module_name, sequence_plan in sequence_parallel_config.items():
        if module_name not in merged:
            merged[module_name] = sequence_plan
            continue

        tp_plan = merged[module_name]
        if not isinstance(tp_plan, dict) or not isinstance(sequence_plan, dict):
            raise TypeError(f"Cannot merge TP and sequence-parallel plans for {module_name}: both plans must be dicts.")
        merged[module_name] = {**tp_plan, **sequence_plan}
    return merged


def _resolve_tp_module_postprocessors(
    model: nn.Module,
    attr_name: str = "TP_MODULE_POSTPROCESSORS",
) -> dict[str, TPModulePostprocessor]:
    """Load optional tensor-parallel postprocessors from the model class."""
    cls = model.__class__
    if not hasattr(cls, attr_name):
        return {}

    postprocessors = getattr(cls, attr_name)
    if postprocessors is None:
        return {}
    if not isinstance(postprocessors, dict):
        raise TypeError(f"{cls.__name__}.{attr_name} must be dict, got {type(postprocessors)}.")

    for module_name, postprocess in postprocessors.items():
        if not callable(postprocess):
            raise TypeError(f"{cls.__name__}.{attr_name}[{module_name!r}] must be callable, got {type(postprocess)}.")

    return postprocessors


def parallelize_model_with_tensor_parallel(
    model: nn.Module,
    tp_mesh,
    sequence_parallel: bool = False,
):
    """Apply tensor parallelism to model submodules declared on the model class.

    The top-level model class must define ``TP_MODULE_CONFIG`` as a mapping from
    runtime module class name to a child-module plan. Each child-module plan maps
    direct child names to ``"col"`` or ``"row"``. When ``sequence_parallel`` is
    enabled, plans may also include ``"sequence"`` entries and may be extended by
    the optional ``SEQUENCE_PARALLEL_MODULE_CONFIG`` class attribute.

    Optional ``TP_MODULE_POSTPROCESSORS`` may map runtime module class names to
    callables that update module-local metadata after tensor sharding.

    Args:
        model: Model instance whose class exposes the tensor-parallel plan.
        tp_mesh: Tensor-parallel device mesh passed to ``parallelize_module``.
        sequence_parallel: Whether to use sequence-parallel layouts for TP activations.
    """
    module_config = _resolve_tp_module_config(model)
    module_postprocessors = _resolve_tp_module_postprocessors(model)
    sequence_dim = 1
    module_sequence_dims: dict[str, int] = {}

    if sequence_parallel:
        sequence_dim = _resolve_sequence_parallel_sequence_dim(model)
        module_sequence_dims = _resolve_sequence_parallel_module_sequence_dims(model)
        sequence_parallel_config = _resolve_tp_module_config(
            model,
            attr_name="SEQUENCE_PARALLEL_MODULE_CONFIG",
            required=False,
        )
        module_config = _merge_tp_module_configs(module_config, sequence_parallel_config)

    applied: list[tuple[str, str, list[str]]] = []
    applied_counts: dict[str, int] = {}
    seen_ids: set[int] = set()
    sequence_parallel_grad_scale_names: set[str] = set()
    sequence_parallel_grad_scale = 1.0 / int(tp_mesh.size()) if sequence_parallel and int(tp_mesh.size()) > 1 else 1.0

    modules = list(model.named_modules())
    modules.sort(key=lambda x: x[0].count("."), reverse=True)

    for path, module in modules:
        if id(module) in seen_ids:
            continue
        cls_name = module.__class__.__name__
        if cls_name not in module_config:
            continue

        plan = _build_tp_plan(
            module_config[cls_name],
            sequence_parallel=sequence_parallel,
            sequence_dim=module_sequence_dims.get(cls_name, sequence_dim),
        )
        if not plan:
            continue

        parallelize_module(module, tp_mesh, plan)
        if sequence_parallel_grad_scale != 1.0:
            prefix = f"{path}." if path else ""
            for parameter_name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    sequence_parallel_grad_scale_names.add(f"{prefix}{parameter_name}")
        postprocess = module_postprocessors.get(cls_name)
        if postprocess is not None:
            postprocess(module, tp_mesh)
        applied.append((path, cls_name, list(plan.keys())))
        applied_counts[cls_name] = applied_counts.get(cls_name, 0) + 1
        seen_ids.add(id(module))

    logger.info("Tensor Parallel Sharding:")
    for cls_name in sorted(applied_counts):
        plan_cfg = module_config[cls_name]
        plan_summary = ", ".join(f"{child}:{mode}" for child, mode in plan_cfg.items())
        logger.info(f"  - ✓ {cls_name} x{applied_counts[cls_name]} ({plan_summary})")
    logger.info(f"  - ✓ {model.__class__.__name__} ({len(applied)} modules)")

    if sequence_parallel_grad_scale_names:
        model._tp_sequence_parallel_grad_scale = sequence_parallel_grad_scale
        model._tp_sequence_parallel_grad_scale_names = tuple(sorted(sequence_parallel_grad_scale_names))
