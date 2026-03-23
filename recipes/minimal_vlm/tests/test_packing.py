from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
import torch
from mvp_dataset.core import RuntimeContext
from omegaconf import OmegaConf

from recipes.minimal_vlm.dataset import PackedSampleAssembler, build_dataset


def _sample(
    tokens: list[int],
    *,
    labels: list[int] | None = None,
    image_count: int = 0,
) -> dict[str, torch.Tensor]:
    sample = {
        "input_ids": torch.tensor(tokens),
        "attention_mask": torch.ones(len(tokens), dtype=torch.long),
        "labels": torch.tensor(labels if labels is not None else tokens),
    }
    if image_count > 0:
        sample["pixel_values"] = torch.randn(image_count, 3, 4, 4)
        sample["image_grid_thw"] = torch.ones(image_count, 3, dtype=torch.long)
    return sample


def _write_jsonl(path: Path, records: list[dict]) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_assembler(
    assembler: PackedSampleAssembler, samples: list[dict[str, torch.Tensor]]
) -> list[dict[str, torch.Tensor]]:
    emitted = []
    for sample in samples:
        emitted.extend(assembler.push(sample))
    emitted.extend(assembler.finish())
    return emitted


class DummyProcessor:
    def apply_chat_template(self, conversations, **kwargs):
        messages = conversations[0]
        message_count = len(messages)
        add_generation_prompt = kwargs["add_generation_prompt"]

        if message_count == 1 and add_generation_prompt:
            return {"input_ids": torch.tensor([[30, 31]])}
        if message_count == 1 and not add_generation_prompt:
            return {
                "input_ids": torch.tensor([[30, 31, 32]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        if message_count == 2 and not add_generation_prompt:
            return {
                "input_ids": torch.tensor([[10, 11, 12, 13]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                "pixel_values": torch.randn(1, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        raise AssertionError(
            f"Unexpected call: message_count={message_count}, add_generation_prompt={add_generation_prompt}"
        )


def test_packed_sample_assembler_best_fit_packs_samples() -> None:
    assembler = PackedSampleAssembler(
        max_length=8,
        selection_strategy="best_fit",
        open_pack_limit=2,
        pack_buffer_size=4,
        seed=0,
    )

    emitted = []
    emitted.extend(assembler.push(_sample([1, 2, 3, 4], image_count=1)))
    emitted.extend(assembler.push(_sample([5, 6, 7])))
    emitted.extend(assembler.finish())

    assert len(emitted) == 1
    packed = emitted[0]
    assert torch.equal(packed["input_ids"], torch.tensor([1, 2, 3, 4, 5, 6, 7]))
    assert torch.equal(packed["attention_mask"], torch.tensor([1, 1, 1, 1, 1, 1, 1]))
    assert torch.equal(packed["labels"], torch.tensor([1, 2, 3, 4, 5, 6, 7]))
    assert packed["pixel_values"].shape == (1, 3, 4, 4)
    assert torch.equal(packed["image_grid_thw"], torch.tensor([[1, 1, 1]]))


def test_packed_sample_assembler_random_is_deterministic_with_seed() -> None:
    samples = [
        _sample([1, 2, 3, 4]),
        _sample([10, 11, 12]),
        _sample([20, 21, 22]),
        _sample([30, 31]),
    ]

    assembler_a = PackedSampleAssembler(
        max_length=8,
        selection_strategy="random",
        open_pack_limit=2,
        pack_buffer_size=8,
        seed=7,
    )
    assembler_b = PackedSampleAssembler(
        max_length=8,
        selection_strategy="random",
        open_pack_limit=2,
        pack_buffer_size=8,
        seed=7,
    )

    outputs_a = _run_assembler(assembler_a, samples)
    outputs_b = _run_assembler(assembler_b, samples)

    assert len(outputs_a) == len(outputs_b)
    assert [output["input_ids"].tolist() for output in outputs_a] == [
        output["input_ids"].tolist() for output in outputs_b
    ]


def test_packed_sample_assembler_buffer_pool_improves_global_packing() -> None:
    samples = [
        _sample([1, 2]),
        _sample([3, 4]),
        _sample([10, 11, 12, 13, 14, 15, 16, 17]),
        _sample([20, 21, 22, 23, 24, 25, 26, 27]),
    ]

    streaming = PackedSampleAssembler(
        max_length=10,
        selection_strategy="best_fit",
        open_pack_limit=2,
        pack_buffer_size=-1,
        seed=0,
    )
    pooled = PackedSampleAssembler(
        max_length=10,
        selection_strategy="best_fit",
        open_pack_limit=2,
        pack_buffer_size=4,
        seed=0,
    )

    streaming_outputs = _run_assembler(streaming, samples)
    pooled_outputs = _run_assembler(pooled, samples)

    assert [output["input_ids"].numel() for output in streaming_outputs] == [8, 4, 8]
    assert [output["input_ids"].numel() for output in pooled_outputs] == [10, 10]


def test_build_dataset_assembles_packed_samples(tmp_path: Path) -> None:
    pytest.importorskip("mvp_dataset")

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "1.jpg").write_bytes(b"test")

    dataset_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "messages": [{"role": "assistant", "content": "Standalone answer."}],
                "images": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "<image>Who is this?"},
                    {"role": "assistant", "content": "An example."},
                ],
                "images": ["images/1.jpg"],
            },
        ],
    )

    config = OmegaConf.create(
        {
            "project": {"dir": str(tmp_path / "outputs")},
            "seed": 42,
            "data": {
                "train_path": str(dataset_path),
                "num_workers": 0,
                "jsonl_num_shards": 1,
                "shuffle_buffer": 1,
                "packing": True,
                "packing_selection_strategy": "best_fit",
                "packing_open_pack_limit": 2,
                "packing_buffer_size": -1,
                "max_seq_len": 8,
            },
        }
    )

    dataset = build_dataset(config, processor=DummyProcessor())
    packed_samples = list(islice(dataset, 1))

    assert len(packed_samples) == 1
    packed = packed_samples[0]
    packed_input_ids = packed["input_ids"].tolist()
    packed_labels = packed["labels"].tolist()
    assert packed_input_ids in (
        [30, 31, 32, 10, 11, 12, 13],
        [10, 11, 12, 13, 30, 31, 32],
    )
    assert packed_labels in (
        [30, 31, 32, -100, -100, 12, 13],
        [-100, -100, 12, 13, 30, 31, 32],
    )
    assert packed["pixel_values"].shape == (1, 3, 4, 4)


def test_packed_sample_assembler_uses_context_seed_for_random_strategy() -> None:
    context = RuntimeContext(seed=11)
    assembler_a = PackedSampleAssembler(
        max_length=8,
        selection_strategy="random",
        open_pack_limit=2,
        pack_buffer_size=4,
        seed=context.sample_shuffle_seed,
    )
    assembler_b = PackedSampleAssembler(
        max_length=8,
        selection_strategy="random",
        open_pack_limit=2,
        pack_buffer_size=4,
        seed=context.sample_shuffle_seed,
    )

    samples = [
        _sample([1, 2, 3]),
        _sample([10, 11, 12]),
        _sample([20, 21]),
        _sample([30, 31]),
    ]

    outputs_a = _run_assembler(assembler_a, samples)
    outputs_b = _run_assembler(assembler_b, samples)

    assert [output["input_ids"].tolist() for output in outputs_a] == [
        output["input_ids"].tolist() for output in outputs_b
    ]
