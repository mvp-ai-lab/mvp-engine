"""Unit tests for the Qwen3 pretraining data logic (kit.llm + recipe packing)."""

import torch

from mvp_engine.kit.llm.data.data import LLMCollator, LLMDataKit, TokenizeAssembler
from recipes.qwen3_pt.model.packing.prepare import build_packed_text_position_ids


class _FakeTokenizer:
    """Minimal tokenizer stub: one token per character, fixed EOS id."""

    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def test_kit_llm_imports():
    """The text-LM kit exposes the public symbols recipes rely on."""
    from mvp_engine.kit.llm import (  # noqa: F401
        LLMCollator,
        LLMDataKit,
        LLMModelKit,
        ModelInputs,
        PackingOptions,
        build_packed_block_causal_mask,
    )


def test_packing_boundary_mask_and_effective_tokens():
    """Each packed document's first label is masked; effective = sum(doc_len - 1)."""
    data_kit = LLMDataKit()
    samples = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "labels": torch.tensor([1, 2, 3]),
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "labels": torch.tensor([4, 5]),
        },
    ]
    packed = data_kit.finalize_packed_samples(samples)
    assert packed["pack_segment_ids"].tolist() == [1, 1, 1, 2, 2]
    assert packed["labels"].tolist() == [-100, 2, 3, -100, 5]

    collated = LLMCollator(pad_token_id=0)([packed])
    assert collated["total_tokens"] == 5
    assert collated["effective_tokens"] == 3  # (3 - 1) + (2 - 1)


def test_long_doc_chunking():
    """Overlong documents are split into <= max_length chunks; blanks are dropped."""
    assembler = TokenizeAssembler(_FakeTokenizer(), max_length=4, text_field="data")
    chunks = assembler.push({"data": "abcdefg"})  # 7 ids + EOS = 8 -> two chunks of 4

    assert len(chunks) == 2
    assert all(chunk["input_ids"].size(0) <= 4 for chunk in chunks)
    assert chunks[-1]["input_ids"][-1].item() == 99  # EOS at the document end
    assert all(torch.equal(chunk["labels"], chunk["input_ids"]) for chunk in chunks)
    assert assembler.push({"data": "   "}) == []


def test_position_ids_reset_per_document():
    """Position ids restart at 0 for each packed document; padding stays 0."""
    segment_ids = torch.tensor([[1, 1, 1, 2, 2, 0]])
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 0]])
    position_ids = build_packed_text_position_ids(input_ids, segment_ids)
    assert position_ids.tolist() == [[0, 1, 2, 0, 1, 0]]
