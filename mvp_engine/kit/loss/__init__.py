"""Loss accounting kits."""

# ruff: noqa: F401

from typing import TYPE_CHECKING

from mvp_engine.kit._lazy import resolve_lazy_export

if TYPE_CHECKING:
    from .loss import LossKit
    from .token_loss import TokenNormedLossKit

_KIT_MODULES = {
    "LossKit": ".loss",
    "TokenNormedLossKit": ".token_loss",
}

__all__ = list(_KIT_MODULES)


def __getattr__(name: str):
    return resolve_lazy_export(globals(), _KIT_MODULES, name)
