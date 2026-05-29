"""Fast step-inference helpers for the PanguVL recipe."""

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from recipes.panguvl.dataset import dataset as dataset_module
from recipes.panguvl.dataset.packing import PackedLengthAssembler, PackedSampleAssembler


class _DummyLogger:
    def info(self, message):
        del message

    def warning(self, message):
        del message


@pytest.fixture(autouse=True)
def _patch_dataset_logger(monkeypatch):
    monkeypatch.setattr(dataset_module, "logger", _DummyLogger())
    dataset_module._LIGHTWEIGHT_SIZE_SOURCE_LOGGED.clear()


class _DummyImageProcessor:
    merge_size = 2

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def get_number_of_image_patches(self, height: int, width: int, options: dict):
        del options
        self.calls.append((height, width))
        return 8


class _DummyTokenizer:
    bos_token = None

    def __call__(self, prompt: str, **kwargs):
        del kwargs
        length = max(1, len(prompt.split()))
        return {
            "input_ids": torch.arange(length, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }


class _DummyProcessor:
    image_token = "<image>"

    def __init__(self):
        self.image_processor = _DummyImageProcessor()
        self.tokenizer = _DummyTokenizer()

    def apply_chat_template(self, messages, *, tokenize=False, **kwargs):
        del kwargs
        assert tokenize is False
        parts: list[str] = []
        for message in messages:
            for block in message["content"]:
                if block["type"] == "image":
                    parts.append(self.image_token)
                elif block["type"] == "text":
                    parts.extend(block["text"].split())
        return " ".join(parts)


def _sample(*, images=None, img_size=None):
    value = {
        "__file__": __file__,
        "__index_in_file__": 0,
        "images": ["unused.jpg"] if images is None else images,
        "conversations": [
            {"from": "human", "value": "<image> describe"},
            {"from": "gpt", "value": "answer"},
        ],
    }
    if img_size is not None:
        value["img_size"] = img_size
    return value


def test_lightweight_process_sample_uses_img_size_without_opening_images(monkeypatch):
    """Stored dimensions should avoid opening image paths or payloads."""
    processor = _DummyProcessor()

    def fail_open(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Image.open should not be called when img_size is present")

    monkeypatch.setattr(dataset_module.Image, "open", fail_open)

    output = dataset_module.lightweight_process_sample(
        _sample(images=[b"not an image"], img_size=[[600, 600]]),
        processor=processor,
        max_length=128,
    )

    assert int(output["input_ids"].numel()) > 0
    assert processor.image_processor.calls == [(600, 600)]


def test_lightweight_process_sample_supports_multiple_img_sizes(monkeypatch):
    """Every image metadata row should feed the multimodal length expansion."""
    processor = _DummyProcessor()
    sample = _sample(
        images=["first.jpg", "second.jpg"],
        img_size=[[600, 600], [720, 960]],
    )
    sample["conversations"][0]["value"] = "<image> first <image> second"
    monkeypatch.setattr(dataset_module.Image, "open", lambda *args, **kwargs: pytest.fail("unexpected image open"))

    output = dataset_module.lightweight_process_sample(sample, processor=processor, max_length=128)

    assert int(output["input_ids"].numel()) > 0
    assert processor.image_processor.calls == [(600, 600), (720, 960)]


def test_lightweight_process_sample_falls_back_when_img_size_missing(tmp_path):
    """Datasets without img_size should still use the existing image probing path."""
    from PIL import Image

    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (17, 13), color=0).save(image_path)

    processor = _DummyProcessor()
    output = dataset_module.lightweight_process_sample(
        _sample(images=[str(image_path)]),
        processor=processor,
        max_length=128,
    )

    assert int(output["input_ids"].numel()) > 0
    assert processor.image_processor.calls == [(13, 17)]


def test_lightweight_process_sample_skips_malformed_img_size():
    """Present but invalid metadata should skip the sample instead of probing images."""
    output = dataset_module.lightweight_process_sample(
        _sample(images=["unused.jpg"], img_size=[["bad", 600]]),
        processor=_DummyProcessor(),
        max_length=128,
    )

    assert output["input_ids"].numel() == 0
    assert output["attention_mask"].numel() == 0


def test_lightweight_process_sample_skips_mismatched_img_size_count():
    """Image metadata count must match the declared image list."""
    output = dataset_module.lightweight_process_sample(
        _sample(images=["first.jpg", "second.jpg"], img_size=[[600, 600]]),
        processor=_DummyProcessor(),
        max_length=128,
    )

    assert output["input_ids"].numel() == 0


def test_unbatched_drop_last_count_matches_old_batched_count_semantics():
    """Counting unbatched packed samples should preserve dataloader drop_last semantics."""
    packed_samples = [{"input_ids": torch.ones(i + 1, dtype=torch.long)} for i in range(7)]
    batch_size = 3

    old_batched_count = sum(len(packed_samples[index : index + batch_size]) for index in range(0, 6, batch_size))
    new_unbatched_count = (len(packed_samples) // batch_size) * batch_size

    assert old_batched_count == 6
    assert new_unbatched_count == old_batched_count


def test_build_step_count_dataset_projects_out_images_column(monkeypatch, tmp_path):
    """The fast step-count path must not request heavy image bytes from parquet."""
    parquet_path = tmp_path / "stage2-like.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "images": [{"bytes": b"heavy-image-bytes", "path": None}],
                    "conversations": [{"from": "human", "value": "<image>"}],
                    "img_size": [[600, 600]],
                }
            ]
        ),
        parquet_path,
    )

    calls: dict[str, object] = {}

    class _FakeDataset:
        @classmethod
        def from_source(cls, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return cls()

        def map(self, fn):
            calls["map_fn"] = fn
            return self

    monkeypatch.setattr(dataset_module, "Dataset", _FakeDataset)

    config = SimpleNamespace(
        seed=7,
        data=SimpleNamespace(
            train_path=str(parquet_path),
            max_seq_len=128,
            enable_thinking=True,
            packing=False,
        ),
    )
    dataset = dataset_module.build_step_count_dataset(config, processor=_DummyProcessor())

    assert isinstance(dataset, _FakeDataset)
    assert calls["kwargs"]["columns"] == ["conversations", "img_size"]
    assert "images" not in calls["kwargs"]["columns"]


@pytest.mark.parametrize("selection_strategy", ["best_fit", "random"])
def test_packed_length_assembler_matches_sample_assembler_count(selection_strategy):
    """Length-only packing should preserve the tensor packer's emitted count."""
    lengths = [3, 9, 4, 7, 12, 1, 5, 6, 2, 8]
    kwargs = {
        "max_length": 12,
        "selection_strategy": selection_strategy,
        "open_pack_limit": 3,
        "pack_buffer_size": 2,
        "seed": 11,
    }
    sample_assembler = PackedSampleAssembler(**kwargs)
    length_assembler = PackedLengthAssembler(**kwargs)

    sample_outputs = []
    length_outputs = []
    for length in lengths:
        sample = {
            "input_ids": torch.ones(length, dtype=torch.long),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "labels": torch.ones(length, dtype=torch.long),
        }
        sample_outputs.extend(sample_assembler.push(sample))
        length_outputs.extend(length_assembler.push({"length": length}))
    sample_outputs.extend(sample_assembler.finish())
    length_outputs.extend(length_assembler.finish())

    assert len(length_outputs) == len(sample_outputs)
    assert [item["length"] for item in length_outputs] == [int(item["input_ids"].size(0)) for item in sample_outputs]
    assert sum(item["sample_count"] for item in length_outputs) == len(lengths)
