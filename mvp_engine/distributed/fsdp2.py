from collections.abc import Callable
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from mvp_engine.utils.log import logger


def _convert_dtype(dtype: Any) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        if not hasattr(torch, dtype):
            raise ValueError(f"Unsupported dtype string: {dtype}")
        return getattr(torch, dtype)
    raise TypeError(f"Unsupported dtype value type: {type(dtype)}")


def _build_mixed_precision_policy(mp_policy: Any) -> MixedPrecisionPolicy | None:
    if mp_policy is None:
        return None
    if isinstance(mp_policy, MixedPrecisionPolicy):
        return mp_policy
    return MixedPrecisionPolicy(
        param_dtype=_convert_dtype(mp_policy.get("param_dtype", None)),
        reduce_dtype=_convert_dtype(mp_policy.get("reduce_dtype", None)),
        output_dtype=_convert_dtype(mp_policy.get("output_dtype", None)),
        cast_forward_inputs=mp_policy.get("cast_forward_inputs", True),
    )


def _resolve_optional_model_callable(model: nn.Module, attr_name: str) -> Callable[[nn.Module], None] | None:
    """Load an optional model-class callable used by runtime customization hooks."""
    cls = model.__class__
    if not hasattr(cls, attr_name):
        return None

    hook = getattr(cls, attr_name)
    if hook is None:
        return None
    if not callable(hook):
        raise TypeError(f"{cls.__name__}.{attr_name} must be callable, got {type(hook)}.")
    return hook


def parallelize_model_with_fsdp2(
    model: nn.Module,
    backend_kwargs: dict = None,
) -> FSDPModule:
    """Parallelize model using FSDP2 (Fully Sharded Data Parallel v2).

    This function applies FSDP2 sharding to specific module types within the model,
    then wraps the entire model with FSDP2. Target modules are determined by the
    model's _no_split_modules attribute or explicitly specified target_classes.

    Args:
        model: The neural network model to parallelize.
        backend_kwargs: Configuration dictionary for FSDP2. Supported keys:
            - target_classes: List of module class names to wrap with FSDP2.
                            Combined with model._no_split_modules if present.
            - high_precision_modules: List of module class names to wrap with high precision
                                    (avoid low precision param/output dtype).
            - high_precision_mp_policy: Optional mixed precision policy for high_precision_modules.
            - ignore_modules: List of module class names to exclude from FSDP2 wrapping.
            - mesh: DeviceMesh for distributed training.
            - reshard_after_forward: FSDP2 resharding strategy.
            - mp_policy: Mixed precision policy.
            Additional kwargs are passed to fully_shard().

    Returns:
        The FSDP2-wrapped model.

    Example:
        >>> model = MyModel()
        >>> kwargs = {"target_classes": ["TransformerBlock"], "mesh": device_mesh}
        >>> fsdp_model = parallelize_model_with_fsdp2(model, kwargs)
    """
    backend_kwargs = dict(backend_kwargs or {})

    # 1. Find all modules to be wrapped with FSDP2.
    #    If a module is selected as target, its whole subtree should not be ignored.
    user_target_classes = backend_kwargs.pop("target_classes", []) or []
    high_precision_classes = set(backend_kwargs.pop("high_precision_modules", []) or [])
    high_precision_mp_policy = _build_mixed_precision_policy(backend_kwargs.pop("high_precision_mp_policy", None))
    model_target_classes = getattr(model, "_no_split_modules", []) or []
    target_classes = set(model_target_classes) | set(user_target_classes)
    target_modules: List[Tuple[str, nn.Module]] = []
    high_precision_modules: List[Tuple[str, nn.Module]] = []
    target_fqns = set()

    ignore_classes = set(backend_kwargs.pop("ignore_modules", []) or [])
    ignore_modules: List[Tuple[str, nn.Module]] = []

    default_mp_policy = _build_mixed_precision_policy(backend_kwargs.get("mp_policy"))
    backend_kwargs["mp_policy"] = default_mp_policy
    if (
        high_precision_classes
        and high_precision_mp_policy is None
        and isinstance(default_mp_policy, MixedPrecisionPolicy)
    ):
        high_precision_mp_policy = MixedPrecisionPolicy(
            param_dtype=None,
            reduce_dtype=default_mp_policy.reduce_dtype,
            output_dtype=None,
            cast_forward_inputs=default_mp_policy.cast_forward_inputs,
        )

    named_modules = list(model.named_modules())
    for fqn, mod in named_modules:
        cls_name = mod.__class__.__name__
        if cls_name in high_precision_classes:
            high_precision_modules.append((fqn, mod))
            target_fqns.add(fqn)
        elif cls_name in target_classes:
            target_modules.append((fqn, mod))
            target_fqns.add(fqn)

    logger.info("FSDP2 Sharding:")
    for cls in sorted(target_classes):
        logger.info(f"  - ✓ {cls}")
    for cls in sorted(high_precision_classes):
        logger.info(f"  - ◎ {cls} (high_precision)")

    def is_in_target_subtree(module_fqn: str) -> bool:
        for target_fqn in target_fqns:
            if target_fqn == "":
                return True
            if module_fqn == target_fqn or module_fqn.startswith(f"{target_fqn}."):
                return True
        return False

    for fqn, mod in named_modules:
        if fqn == "":
            continue  # Skip root module; it is handled by the final fully_shard call
        if is_in_target_subtree(fqn):
            continue

        if mod.__class__.__name__ in ignore_classes:
            ignore_modules.append((fqn, mod))

    ignored_params: set[nn.Parameter] = set()
    for fqn, mod in ignore_modules:
        for param in mod.parameters(recurse=False):
            ignored_params.add(param)
        logger.info(f"  - ✗ {mod.__class__.__name__} ({fqn})")

    # 2. Wrap target modules with FSDP2
    high_precision_backend_kwargs = dict(backend_kwargs)
    if high_precision_mp_policy is not None:
        high_precision_backend_kwargs["mp_policy"] = high_precision_mp_policy

    wrap_plan: List[Tuple[str, nn.Module, dict]] = []
    for fqn, mod in target_modules:
        wrap_plan.append((fqn, mod, backend_kwargs))
    for fqn, mod in high_precision_modules:
        wrap_plan.append((fqn, mod, high_precision_backend_kwargs))

    wrap_plan.sort(key=lambda item: item[0].count("."), reverse=True)
    for _fqn, mod, shard_kwargs in wrap_plan:
        fully_shard(mod, **shard_kwargs)

    # 3. Wrap the whole model with FSDP2
    fully_shard(model, **backend_kwargs, ignored_params=ignored_params)
    logger.info(f"  - ✓ {model.__class__.__name__} (entire model)")

    # 4. Apply FSDP2 prefetching if needed
    apply_fsdp2_custom_prefetching = _resolve_optional_model_callable(model, "APPLY_FSDP2_CUSTOM_PREFETCHING")
    if apply_fsdp2_custom_prefetching is not None:
        logger.info(f"Applying custom FSDP2 prefetching on {model.__class__.__name__}...")
        apply_fsdp2_custom_prefetching(model)

    return model
