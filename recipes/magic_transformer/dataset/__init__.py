"""Dataset helpers for the Magic Transformer recipe."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset import FakeAutoregressiveDataset, build_dataset
    from .sampler import InfiniteDistributedSampler

__all__ = ["FakeAutoregressiveDataset", "InfiniteDistributedSampler", "build_dataset"]

_EXPORT_MODULES = {
    "FakeAutoregressiveDataset": ".dataset",
    "InfiniteDistributedSampler": ".sampler",
    "build_dataset": ".dataset",
}


def __getattr__(name: str):
    """Lazily resolve Magic Transformer dataset exports."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
