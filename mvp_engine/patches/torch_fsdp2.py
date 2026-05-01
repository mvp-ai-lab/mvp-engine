"""Runtime patches for PyTorch FSDP2."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType
from typing import Any

from .types import PatchResult

_PATCH_NAME = "torch_fsdp2_checkpoint_recompute_cast"


def _load_fsdp_state_module() -> ModuleType | None:
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_state as fsdp_state
    except Exception:
        return None
    return fsdp_state


def _has_required_symbols(fsdp_state: ModuleType) -> bool:
    return all(
        hasattr(fsdp_state, name)
        for name in (
            "FSDPState",
            "FSDPParamGroup",
            "TrainingState",
            "_apply_to_tensors",
            "_cast_fp_tensor",
            "disable_if_config_true",
        )
    )


def _is_legacy_pre_forward(source: str) -> bool:
    normalized = "\n".join(line.rstrip() for line in source.splitlines())
    return (
        "        if self._training_state == TrainingState.PRE_BACKWARD:\n            return args, kwargs"
    ) in normalized


def apply_fsdp2_checkpoint_recompute_cast_patch(
    fsdp_state: ModuleType | None = None,
) -> PatchResult:
    """Restore FSDP2 mixed-precision input casting during checkpoint recompute.

    Older PyTorch FSDP2 returns early from ``FSDPState._pre_forward`` while the
    state is ``PRE_BACKWARD``. Activation-checkpoint recompute then skips
    ``cast_forward_inputs``, so the original forward may save bf16 metadata while
    recompute produces fp32 tensors. This mirrors the upstream fix by only
    applying input casting on that early-return path.
    """

    if fsdp_state is None:
        fsdp_state = _load_fsdp_state_module()
    if fsdp_state is None:
        return PatchResult(_PATCH_NAME, "skipped", "PyTorch FSDP2 module is unavailable")
    if not _has_required_symbols(fsdp_state):
        return PatchResult(_PATCH_NAME, "skipped", "PyTorch FSDP2 module shape is unsupported")

    fsdp_state_cls = fsdp_state.FSDPState
    if getattr(fsdp_state_cls, "_mvp_engine_recompute_cast_patch_applied", False):
        return PatchResult(_PATCH_NAME, "skipped", "already applied")

    try:
        source = inspect.getsource(fsdp_state_cls._pre_forward)
    except (OSError, TypeError):
        return PatchResult(_PATCH_NAME, "skipped", "cannot inspect FSDPState._pre_forward")

    if "return self._mvp_engine_cast_forward_inputs(args, kwargs)" in source:
        return PatchResult(_PATCH_NAME, "skipped", "already patched")
    if "return self._cast_forward_inputs(args, kwargs)" in source:
        return PatchResult(_PATCH_NAME, "skipped", "upstream fix appears present")
    if not _is_legacy_pre_forward(source):
        return PatchResult(_PATCH_NAME, "skipped", "FSDPState._pre_forward is not a known legacy shape")

    def _mvp_engine_cast_forward_inputs(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            import torch

            with torch.profiler.record_function("FSDP::cast_forward_inputs"):
                cast_fn = functools.partial(
                    fsdp_state._cast_fp_tensor,
                    self._mp_policy.param_dtype,
                )
                args, kwargs = (
                    fsdp_state._apply_to_tensors(cast_fn, args),
                    fsdp_state._apply_to_tensors(cast_fn, kwargs),
                )
        return args, kwargs

    @fsdp_state.disable_if_config_true
    def _pre_forward(self: Any, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        # During activation-checkpoint recompute, pre-backward already handled
        # unshard. Only restore the mixed-precision input cast that legacy
        # PyTorch skipped via the early return.
        if self._training_state == fsdp_state.TrainingState.PRE_BACKWARD:
            return self._mvp_engine_cast_forward_inputs(args, kwargs)

        self._training_state = fsdp_state.TrainingState.FORWARD
        args, kwargs = self._root_pre_forward(module, args, kwargs)
        args, kwargs = self._mvp_engine_cast_forward_inputs(args, kwargs)
        if self._fsdp_param_group:
            args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
        for state in self._states_to_forward_prefetch:
            if (target_param_group := state._fsdp_param_group) is not None:
                fsdp_state.FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
        return args, kwargs

    fsdp_state_cls._mvp_engine_original_pre_forward = fsdp_state_cls._pre_forward
    fsdp_state_cls._mvp_engine_cast_forward_inputs = _mvp_engine_cast_forward_inputs
    fsdp_state_cls._pre_forward = _pre_forward
    fsdp_state_cls._mvp_engine_recompute_cast_patch_applied = True

    return PatchResult(_PATCH_NAME, "applied", "patched FSDPState._pre_forward recompute input casting")
