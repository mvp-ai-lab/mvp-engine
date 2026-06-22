"""Unit tests for the Qwen3 pretrain data logic (kit.llm + recipe packing)."""

import torch

from mvp_engine.kit import (
    LLMBatchCollator,
    LLMPackingAssembler,
    LLMPackingSpec,
    LLMPretrainTextSchemaHandler,
    LLMPretrainTextTokenizationHandler,
    LLMSample,
    LLMSampleSpec,
    LLMStepEstimationKit,
    QwenChatSchemaHandler,
    QwenChatTokenizationHandler,
)
from recipes.qwen3.model.packing.prepare import build_packed_text_position_ids


class _FakeTokenizer:
    """Minimal tokenizer stub: one token per character, fixed EOS id."""

    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self(text, add_special_tokens=False)["input_ids"] if tokenize else text


class _FakeConsumableDataset:
    """Minimal mvp-dataset consume() stub for step estimation tests."""

    def __init__(self, sample_count):
        self.sample_count = sample_count

    def consume(self, factory):
        consumer = factory(None)
        for _ in range(self.sample_count):
            if not consumer.push({}):
                break
        return consumer.finish()


def test_kit_llm_imports():
    """The text-LM kit exposes the public symbols recipes rely on."""
    from mvp_engine.kit.llm import (  # noqa: F401
        LLMBatchCollator,
        LLMDataKit,
        LLMModelKit,
        LLMPackingAssembler,
        LLMPackingSpec,
        LLMPretrainTextSchemaHandler,
        LLMPretrainTextTokenizationHandler,
        LLMStepEstimateResult,
        LLMStepEstimationKit,
        ModelInputs,
        QwenChatSchemaHandler,
        QwenChatTokenizationHandler,
        build_packed_block_causal_mask,
    )


def test_stream_packing_concatenates_samples_and_keeps_boundary_labels():
    """Sequential packing concatenates text samples into fixed-length token streams."""
    sample_spec = LLMSampleSpec(
        schema_handler=LLMPretrainTextSchemaHandler(),
        tokenization_handler=LLMPretrainTextTokenizationHandler(tokenizer=_FakeTokenizer(), max_seq_len=8),
    )
    samples = [
        LLMSample.from_tokens(
            {"data": "abc"},
            sample_spec=sample_spec,
            metadata={},
            input_ids=[1, 2, 3],
            labels=[1, 2, 3],
            attention_mask=[1, 1, 1],
        ),
        LLMSample.from_tokens(
            {"data": "de"},
            sample_spec=sample_spec,
            metadata={},
            input_ids=[4, 5],
            labels=[4, 5],
            attention_mask=[1, 1],
        ),
    ]
    assembler = LLMPackingAssembler(LLMPackingSpec(max_seq_len=5))
    packed = []
    for sample in samples:
        packed.extend(assembler.push(sample))

    assert len(packed) == 1
    model_inputs = packed[0].to_model_inputs()
    assert model_inputs["input_ids"].tolist() == [1, 2, 3, 4, 5]
    assert model_inputs["pack_segment_ids"].tolist() == [1, 1, 1, 1, 1]
    assert model_inputs["labels"].tolist() == [1, 2, 3, 4, 5]

    collated = LLMBatchCollator(pad_token_id=0)([model_inputs])
    assert collated["total_tokens"] == 5
    assert collated["effective_tokens"] == 4


def test_long_doc_chunking():
    """Overlong documents are split into <= max_length chunks; blanks are dropped."""
    sample_spec = LLMSampleSpec(
        schema_handler=LLMPretrainTextSchemaHandler(text_field="data"),
        tokenization_handler=LLMPretrainTextTokenizationHandler(tokenizer=_FakeTokenizer(), max_seq_len=4),
    )
    chunks = LLMSample.from_row({"data": "abcdefg"}, sample_spec=sample_spec).to_chunks()

    assert len(chunks) == 2
    assert all(chunk.token_length <= 4 for chunk in chunks)
    assert chunks[-1].input_ids[-1] == 99  # EOS at the document end
    assert all(chunk.labels == chunk.input_ids for chunk in chunks)
    assert LLMSample.from_row({"data": "   "}, sample_spec=sample_spec).to_chunks() == []


def test_position_ids_are_isolated_per_document():
    """Position ids start at 0 for each packed stream segment; padding stays 0."""
    segment_ids = torch.tensor([[1, 1, 1, 2, 2, 0]])
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 0]])
    position_ids = build_packed_text_position_ids(input_ids, segment_ids)
    assert position_ids.tolist() == [[0, 1, 2, 0, 1, 0]]


def test_qwen_chat_schema_marks_only_assistant_target_for_loss():
    """Qwen chat schema renders prompt text as source and assistant text as target."""
    tokenizer = _FakeTokenizer()
    sample_spec = LLMSampleSpec(
        schema_handler=QwenChatSchemaHandler(tokenizer=tokenizer, thinking_mode=False),
        tokenization_handler=QwenChatTokenizationHandler(tokenizer=tokenizer, max_seq_len=256, add_eos=False),
    )
    chunks = LLMSample.from_row(
        {
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "<think>\n\n</think>\n\nhello"},
            ]
        },
        sample_spec=sample_spec,
    ).to_chunks()

    assert len(chunks) == 1
    labels = chunks[0].labels
    supervised_start = next(index for index, label in enumerate(labels) if label != -100)
    target_text = "hello<|im_end|>\n"
    assert len(labels) - supervised_start == len(target_text)
    assert all(label == -100 for label in labels[:supervised_start])


def test_llm_step_estimation_counts_packed_samples_exactly():
    """LLM step estimation counts finite packed samples exactly."""
    result = LLMStepEstimationKit().estimate_total_steps(
        _FakeConsumableDataset(sample_count=5),
        batch_size=2,
        gradient_accumulation_steps=1,
        data_parallel_world_size=1,
    )

    assert result.total_steps == 3
    assert result.total_packed_samples == 5
    assert result.samples_per_step == 2
    assert result.exact is True
