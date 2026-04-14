import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _install_test_stubs() -> None:
    if "mvp_dataset" not in sys.modules:
        mvp_dataset = types.ModuleType("mvp_dataset")
        mvp_dataset.Dataset = object
        mvp_dataset.TorchLoader = object
        mvp_dataset.set_logger = lambda *_args, **_kwargs: None
        sys.modules["mvp_dataset"] = mvp_dataset

    if "mvp_dataset.core" not in sys.modules:
        core = types.ModuleType("mvp_dataset.core")

        class RuntimeContext:
            @staticmethod
            def from_runtime(seed):
                return SimpleNamespace(sample_shuffle_seed=seed)

        class Assembler:
            def __class_getitem__(cls, _item):
                return cls

        core.RuntimeContext = RuntimeContext
        core.Assembler = Assembler
        sys.modules["mvp_dataset.core"] = core

    if "mvp_dataset.utils" not in sys.modules:
        sys.modules["mvp_dataset.utils"] = types.ModuleType("mvp_dataset.utils")

    if "mvp_dataset.utils.url" not in sys.modules:
        url = types.ModuleType("mvp_dataset.utils.url")
        url.normalize_paths = lambda path: [path]
        sys.modules["mvp_dataset.utils.url"] = url

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForImageTextToText = object
        transformers.AutoProcessor = object
        sys.modules["transformers"] = transformers

    if "transformers.utils" not in sys.modules:
        sys.modules["transformers.utils"] = types.ModuleType("transformers.utils")

    if "transformers.utils.logging" not in sys.modules:
        logging_mod = types.ModuleType("transformers.utils.logging")
        logging_mod.disable_progress_bar = lambda: None
        sys.modules["transformers.utils.logging"] = logging_mod


_install_test_stubs()
openbee_dataset = importlib.import_module("recipes.openbee.dataset.dataset")
openbee_collator = importlib.import_module("recipes.openbee.dataset.collator")
openbee_packing = importlib.import_module("recipes.openbee.dataset.packing")
engine_mod = importlib.import_module("mvp_engine.engine.engine")

Engine = engine_mod.Engine
OpenbeeCollator = openbee_collator.OpenbeeCollator
PackedSampleAssembler = openbee_packing.PackedSampleAssembler
build_skipped_sample = openbee_dataset.build_skipped_sample
process_sample = openbee_dataset.process_sample


def test_process_sample_skips_invalid_image_with_warning(monkeypatch, tmp_path):
    sample_path = tmp_path / "train-00000-of-00001.parquet"
    processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: None)

    def _raise_broken_png(*_args, **_kwargs):
        raise SyntaxError("broken PNG file (chunk b'WU\\x95\\xe3')")

    monkeypatch.setattr(openbee_dataset, "process_image", _raise_broken_png)

    sample = {
        "__file__": str(sample_path),
        "__index_in_file__": 6428,
        "messages": [{"role": "user", "content": "<image>"}],
        "images": [{"bytes": b"bad"}],
    }

    with pytest.warns(RuntimeWarning, match=r"Skipping invalid OpenBee sample .*broken PNG file"):
        processed = process_sample(sample, processor=processor, max_length=128)

    assert processed["input_ids"].numel() == 0
    assert processed["attention_mask"].numel() == 0
    assert processed["labels"].numel() == 0


def test_packed_sample_assembler_ignores_skipped_samples():
    assembler = PackedSampleAssembler(max_length=8)

    assert list(assembler.push(build_skipped_sample())) == []
    assert list(assembler.finish()) == []


def test_openbee_collator_filters_skipped_samples():
    collator = OpenbeeCollator(pad_token_id=0)
    valid_sample = {
        "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([-100, 12, 13], dtype=torch.long),
    }

    batch = collator([build_skipped_sample(), valid_sample])

    assert batch is not None
    assert tuple(batch["input_ids"].shape) == (1, 3)
    assert torch.equal(batch["input_ids"][0], valid_sample["input_ids"])
    assert collator([build_skipped_sample()]) is None


def test_run_iter_train_skips_none_batches():
    processed_batches: list[int] = []

    def train_pre_step(data):
        if data == "skip":
            return None
        return data

    def train_one_step(data):
        processed_batches.append(data)
        return {"loss": torch.tensor(0.0), "logs": {}}

    def train_after_step(_outputs):
        engine.step += 1
        return _outputs

    engine = SimpleNamespace(
        step=0,
        total_steps=2,
        train_loader=["skip", 3, 4],
        train_pre_step=train_pre_step,
        train_one_step=train_one_step,
        train_after_step=train_after_step,
    )

    Engine.run_iter_train(engine)

    assert processed_batches == [3, 4]
    assert engine.step == 2
