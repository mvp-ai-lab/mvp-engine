from collections.abc import Callable, Iterable
from typing import Any

import torch.nn as nn

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

_TP_STYLE_FACTORIES = {
    "col": ColwiseParallel,
    "row": RowwiseParallel,
}

TPModulePostprocessor = Callable[[nn.Module, Any], None]


def _normalize_module_path(path: str) -> str:
    """Normalize module paths so user filters can be matched consistently."""
    return path.strip(".")


def _matches_module_path(path: str, prefix: str) -> bool:
    """Return True when ``path`` is the same module as ``prefix`` or its subtree."""
    normalized_path = _normalize_module_path(path)
    normalized_prefix = _normalize_module_path(prefix)
    if normalized_prefix == "":
        return True
    return normalized_path == normalized_prefix or normalized_path.startswith(f"{normalized_prefix}.")


def _mode_to_style(mode: Any, sequence_parallel: bool = False, sequence_dim: int = 1) -> ParallelStyle:
    if isinstance(mode, str):
        mode_key = mode.lower()
        is_sequence_mode = sequence_parallel and mode_key == "sequence"
        if mode_key not in _TP_STYLE_FACTORIES and not is_sequence_mode:
            expected_modes = list(_TP_STYLE_FACTORIES)
            if sequence_parallel:
                expected_modes.append("sequence")
            raise ValueError(f"Unknown TP mode '{mode}'. Expected one of {expected_modes}.")
        if not sequence_parallel:
            return _TP_STYLE_FACTORIES[mode_key]()
        if mode_key == "col":
            return ColwiseParallel(input_layouts=Shard(sequence_dim))
        if mode_key == "row":
            return RowwiseParallel(output_layouts=Shard(sequence_dim))
        if mode_key == "sequence":
            return SequenceParallel(sequence_dim=sequence_dim, use_local_output=True)

    if callable(mode):
        style = mode()
        if not isinstance(style, ParallelStyle):
            raise ValueError(f"Callable TP mode must return ParallelStyle, got {type(style)}.")
        return style

    if isinstance(mode, ParallelStyle):
        return mode

    raise ValueError(f"Unsupported TP mode type: {type(mode)}")


def _build_tp_plan(
    module: nn.Module,
    plan_cfg: object,
    sequence_parallel: bool = False,
    sequence_dim: int = 1,
) -> dict[str, ParallelStyle]:
    if isinstance(plan_cfg, (list, tuple)):
        linear_children = [(name, child) for name, child in module.named_children() if isinstance(child, nn.Linear)]
        if len(plan_cfg) != len(linear_children):
            raise ValueError(
                f"Plan length ({len(plan_cfg)}) does not match linear children ({len(linear_children)}) "
                f"for module {module.__class__.__name__}."
            )
        return {
            name: _mode_to_style(mode, sequence_parallel=sequence_parallel, sequence_dim=sequence_dim)
            for (name, _), mode in zip(linear_children, plan_cfg)
        }

    if not isinstance(plan_cfg, dict):
        raise TypeError(f"Unsupported TP plan config type: {type(plan_cfg)}")

    plan: dict[str, ParallelStyle] = {}
    for child_name, mode in plan_cfg.items():
        plan[child_name] = _mode_to_style(mode, sequence_parallel=sequence_parallel, sequence_dim=sequence_dim)
    return plan


def _should_apply(path: str, include_paths: Iterable[str], exclude_paths: Iterable[str]) -> bool:
    if include_paths and not any(_matches_module_path(path, prefix) for prefix in include_paths):
        return False
    if exclude_paths and any(_matches_module_path(path, prefix) for prefix in exclude_paths):
        return False
    return True


def resolve_tp_module_config(model: nn.Module, attr_name: str = "TP_MODULE_CONFIG") -> dict[str, object]:
    """Load the model-defined tensor-parallel module config from a class attribute."""
    cls = model.__class__
    if not hasattr(cls, attr_name):
        raise AttributeError(
            f"{cls.__name__} does not define class attribute '{attr_name}'. "
            "Please provide TP module plan via model class attribute."
        )

    module_config = getattr(cls, attr_name)
    if module_config is None:
        raise ValueError(f"{cls.__name__}.{attr_name} is None.")
    if not isinstance(module_config, dict):
        raise TypeError(f"{cls.__name__}.{attr_name} must be dict, got {type(module_config)}.")

    return module_config


def resolve_optional_tp_module_config(model: nn.Module, attr_name: str) -> dict[str, object]:
    """Load an optional tensor-parallel module config from a class attribute."""
    cls = model.__class__
    if not hasattr(cls, attr_name):
        return {}

    module_config = getattr(cls, attr_name)
    if module_config is None:
        return {}
    if not isinstance(module_config, dict):
        raise TypeError(f"{cls.__name__}.{attr_name} must be dict, got {type(module_config)}.")

    return module_config


def resolve_sequence_parallel_sequence_dim(
    model: nn.Module,
    attr_name: str = "SEQUENCE_PARALLEL_SEQUENCE_DIM",
) -> int:
    """Load the sequence dimension used by sequence parallel layouts."""
    cls = model.__class__
    sequence_dim = getattr(cls, attr_name, 1)
    if not isinstance(sequence_dim, int):
        raise TypeError(f"{cls.__name__}.{attr_name} must be int, got {type(sequence_dim)}.")
    return sequence_dim


def merge_tp_module_configs(
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


def resolve_tp_module_postprocessors(
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
    include_paths: Iterable[str] = (),
    exclude_paths: Iterable[str] = (),
    sequence_parallel: bool = False,
) -> list[tuple[str, str, list[str]]]:
    """Apply tensor parallelism and optional sequence-parallel layouts to selected submodules."""
    module_config = resolve_tp_module_config(model, attr_name="TP_MODULE_CONFIG")
    module_postprocessors = resolve_tp_module_postprocessors(model)
    sequence_dim = 1

    if sequence_parallel:
        sequence_dim = resolve_sequence_parallel_sequence_dim(model)
        sequence_parallel_config = resolve_optional_tp_module_config(
            model,
            attr_name="SEQUENCE_PARALLEL_MODULE_CONFIG",
        )
        module_config = merge_tp_module_configs(module_config, sequence_parallel_config)

    applied: list[tuple[str, str, list[str]]] = []
    seen_ids: set[int] = set()

    modules = list(model.named_modules())
    modules.sort(key=lambda x: x[0].count("."), reverse=True)

    for path, module in modules:
        if id(module) in seen_ids:
            continue
        cls_name = module.__class__.__name__
        if cls_name not in module_config:
            continue
        if not _should_apply(path, include_paths, exclude_paths):
            continue

        plan = _build_tp_plan(
            module,
            module_config[cls_name],
            sequence_parallel=sequence_parallel,
            sequence_dim=sequence_dim,
        )
        if not plan:
            continue

        parallelize_module(module, tp_mesh, plan)
        postprocess = module_postprocessors.get(cls_name)
        if postprocess is not None:
            postprocess(module, tp_mesh)
        applied.append((path, cls_name, list(plan.keys())))
        seen_ids.add(id(module))

    return applied
