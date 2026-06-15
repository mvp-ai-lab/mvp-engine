"""Padding invariants for the PanguVL loss path."""

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_collator_class():
    """Load the real collator module without importing the full recipe package."""
    recipe_root = Path(__file__).resolve().parent.parent
    package_name = "_panguvl_test_dataset"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(recipe_root / "dataset")]
        sys.modules[package_name] = package

    module_name = f"{package_name}.collator"
    spec = importlib.util.spec_from_file_location(module_name, recipe_root / "dataset" / "collator.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.PanguvlCollator


PanguvlCollator = _load_collator_class()


class _DummyProcessor:
    pass


def _shift_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.pad(labels, (0, 1), value=ignore_index)[..., 1:]


def test_collator_padding_is_ignored_by_shifted_loss_labels():
    """Padded input tokens must never become supervised shifted labels."""
    pad_token_id = 2
    ignore_index = -100
    collator = PanguvlCollator(
        pad_token_id=pad_token_id,
        processor=_DummyProcessor(),
        ignore_index=ignore_index,
    )

    batch = collator(
        [
            {
                "input_ids": torch.tensor([10, 11, 12, 13], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([ignore_index, 11, 12, 13], dtype=torch.long),
                "pixel_values": torch.zeros((1, 3), dtype=torch.float32),
            },
            {
                "input_ids": torch.tensor([20, 21], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                "labels": torch.tensor([ignore_index, 21], dtype=torch.long),
                "pixel_values": torch.zeros((1, 3), dtype=torch.float32),
            },
        ]
    )

    assert batch["input_ids"].tolist() == [
        [10, 11, 12, 13],
        [20, 21, pad_token_id, pad_token_id],
    ]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ]
    assert batch["labels"].tolist() == [
        [ignore_index, 11, 12, 13],
        [ignore_index, 21, ignore_index, ignore_index],
    ]

    shifted_labels = _shift_labels(batch["labels"], ignore_index=ignore_index)
    supervised_shifted = shifted_labels.ne(ignore_index)
    pad_positions = batch["input_ids"].eq(pad_token_id)

    assert not torch.any(batch["labels"][pad_positions].ne(ignore_index))
    assert not torch.any(shifted_labels[supervised_shifted].eq(pad_token_id))
    assert int(supervised_shifted.sum().item()) == 4
