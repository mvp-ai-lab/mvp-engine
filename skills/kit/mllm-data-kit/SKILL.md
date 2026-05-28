---
name: mllm-data-kit
description: Use and extend MVP-Engine MLLM data kits, including MLLMDataKit,
  MLLMSampleKit, MLLMMediaKit, PackingOptions, processor setup, always-on
  packing, media materialization, collation, and multimodal data extensions.
---

# MLLM Data Kit

## Goal

Use `mvp_engine.kit.mllm` data APIs instead of rebuilding recipe-local VLM data
pipelines:

- `MLLMDataKit` orchestrates processor, dataset, chat SFT preprocessing,
  always-on packing, media materialization, collation, dataloader, and
  `to_device`;
- `MLLMSampleKit` normalizes raw rows into canonical chat messages and ordered
  media refs;
- `MLLMMediaKit` implements model-family media token expansion, label masking,
  late media materialization, and batch media collation;
- `PackingOptions` exposes packing strategy knobs while packing remains enabled
  by default in the standard pipeline.

## Required Inputs

Identify these before editing:

- target recipe path and engine data entrypoint;
- raw row schema: messages/conversations, media refs, media sizes, placeholders;
- target processor and model-facing batch fields;
- whether the default Qwen-style image media behavior is sufficient;
- whether new schema or modality support needs a custom `MLLMSampleKit` or
  `MLLMMediaKit`;
- recipe-local structure/smoke tests.

Ask the user only when raw schema, media semantics, or target processor behavior
cannot be derived locally.

## Workflow

### 1. Use The Standard Pipeline First

Prefer this shape in recipe engines:

```python
from mvp_engine.kit import MLLMDataKit, PackingOptions

self.data_kit = MLLMDataKit()
processor = self.data_kit.build_processor(...)
packing = PackingOptions(selection_strategy="best_fit", open_pack_limit=8, buffer_size=64)
dataset = self.data_kit.build_dataset(
    dataset_path=...,
    processor=processor,
    max_seq_len=...,
    ref_columns=("images",),
    packing=packing,
    thinking_mode=True,
)
collate_fn = self.data_kit.build_collator(
    pad_token_id=int(processor.tokenizer.pad_token_id),
    processor=processor,
)
dataloader = self.data_kit.build_dataloader(dataset, batch_size=..., num_workers=..., collate_fn=collate_fn)
```

### 2. Keep Kit Boundaries Clear

- Extend `MLLMSampleKit` when raw rows use different field names, role aliases,
  placeholder strings, or media ordering rules.
- Extend `MLLMMediaKit` when a model family needs different media tokens,
  token-count estimation, truncation rules, media loading, or collated tensor
  fields.
- Extend `MLLMDataKit` only when orchestration changes: dataset backend,
  chat-SFT turn construction, packing lifecycle, guard placement, or dataloader
  policy.

Read `references/sample-kit.md`, `references/media-kit.md`, and
`references/packing.md` before writing custom subclasses.

### 3. Add New Modalities Through SampleKit And MediaKit

For video, audio, or omni data:

- make `MLLMSampleKit.normalize()` emit canonical `CanonicalMedia(type=...)`
  records without reading heavy media;
- make `MLLMMediaKit.prepare()` compute placeholder token counts and sample
  fields from metadata;
- make `render_text()` consume media placeholders in model order;
- make `materialize()` perform expensive IO such as video frame sampling after
  refs are resolved;
- make `collate()` concatenate or pad model-family media tensors.

Do not make `MLLMDataKit` know video codecs, frame sampling policy, audio
windowing, or model-family special tokens.

## Validation

### Soft Validation

Review the changed recipe and kit subclass code:

- standard recipes call `MLLMDataKit` rather than duplicating preprocess,
  packing, and collator code;
- SampleKit owns raw schema normalization only;
- MediaKit owns media token/render/materialize/collate behavior only;
- DataKit owns orchestration only;
- media refs are materialized after `resolve_ref` unless the backend explicitly
  requires eager loading;
- packed samples contain token fields, `pack_segment_ids`, `source_sample_num`,
  and media tensors expected by the model preparation path;
- text-only, single-media, multi-media, and invalid samples have explicit
  behavior.

### Hard Validation

Run the recipe's existing gates:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

For modality extensions, add or update recipe-local skill assertions only when
structure/smoke cannot verify the new contract.

## Output

- State which kit APIs are used.
- State whether custom SampleKit, MediaKit, or DataKit subclasses were added.
- State raw schema, media materialization boundary, packing knobs, and collator
  outputs.
- Report validation commands and any unvalidated modality cases.

## Read On Demand

- `references/sample-kit.md`: raw schema and canonical sample extension rules.
- `references/media-kit.md`: media token, materialization, and collation rules.
- `references/packing.md`: always-on packing and packed metadata rules.
