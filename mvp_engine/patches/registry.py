"""Small runtime patch registry.

Patches in this package must be idempotent and fail closed on unsupported
library versions.
"""

from __future__ import annotations

import os
import warnings
from typing import Callable

from .torch_fsdp2 import apply_fsdp2_checkpoint_recompute_cast_patch
from .types import PatchResult

PatchFn = Callable[[], PatchResult]

_PATCHES: tuple[PatchFn, ...] = (apply_fsdp2_checkpoint_recompute_cast_patch,)
_APPLIED = False


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def apply_all_patches() -> list[PatchResult]:
    """Apply all registered runtime patches once per process."""

    global _APPLIED
    if _APPLIED:
        return []
    _APPLIED = True

    if _env_flag("MVP_ENGINE_DISABLE_RUNTIME_PATCHES"):
        return [
            PatchResult(
                name="runtime_patches",
                status="skipped",
                reason="disabled by MVP_ENGINE_DISABLE_RUNTIME_PATCHES",
            )
        ]

    strict = _env_flag("MVP_ENGINE_STRICT_RUNTIME_PATCHES")
    results: list[PatchResult] = []
    for patch_fn in _PATCHES:
        try:
            result = patch_fn()
        except Exception as exc:
            if strict:
                raise
            result = PatchResult(
                name=getattr(patch_fn, "__name__", patch_fn.__class__.__name__),
                status="failed",
                reason=str(exc),
            )
            warnings.warn(
                f"MVP Engine runtime patch {result.name!r} failed: {result.reason}",
                RuntimeWarning,
                stacklevel=2,
            )
        results.append(result)

    return results
