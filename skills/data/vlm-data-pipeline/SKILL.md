---
name: vlm-data-pipeline
description: Add, review, update, and validate recipe-local VLM data pipelines
  from raw rows to model-ready multimodal batches, including schema checks,
  processor setup, preprocessing, media materialization, collation, guards, and
  optional packing boundaries.
---

# VLM Data Pipeline

## Goal

Build or modify a recipe-local VLM data pipeline that preserves the model's
multimodal contract:

- normalize raw conversations, media references, and media-size metadata;
- render the target model's chat/template format;
- tokenize text and create supervised labels;
- materialize images/videos at the correct lifecycle boundary;
- collate text-only and multimodal samples into valid model batches;
- define invalid-sample and skip behavior explicitly;
- keep packing internals delegated to `skills/data/vlm-packing/` when packing is
  involved.

Use `recipes/basic_vlm` as the reference implementation, not as a required
shape for every recipe.

## Required Inputs

Identify these before editing:

- target recipe path;
- dataset backend and loader lifecycle;
- raw row schema for conversations, media refs, and media sizes;
- target processor, chat template, media placeholder, and output tensor names;
- model-facing batch fields consumed by `forward()`;
- invalid-sample policy and accounting boundary;
- whether packing is enabled;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if the raw schema, target processor, or dataset backend cannot
be derived from the recipe.

## Workflow

### 1. Locate The Existing Pipeline

Search the recipe first:

```bash
rg -n "build_dataset|process_sample|AutoProcessor|apply_chat_template" recipes/<recipe>
rg -n "collator|pixel_values|image_grid_thw|labels|attention_mask|packing" recipes/<recipe>
```

Find:

- where processor construction happens;
- where raw rows become tokenized samples;
- where media references are resolved or decoded;
- where invalid samples are dropped or converted to sentinels;
- where the collator builds the final model batch.

### 2. Identify Backend And Lifecycle

Before copying Basic VLM patterns, identify whether the recipe uses
`mvp_dataset`, PyTorch `Dataset` / `IterableDataset`, Hugging Face `datasets`, or
a custom loader.

Read `references/pipeline_rules.md` when backend lifecycle, materialization
boundary, or skip accounting is unclear.

### 3. Define Raw And Processed Contracts

Document the raw row contract near preprocessing code:

- conversation field and supported roles;
- image/video reference fields;
- media-size fields and ordering;
- placeholder count rules;
- optional metadata preserved for debugging.

Document the processed sample contract:

- `input_ids`;
- `attention_mask`;
- `labels`;
- optional `pixel_values`, `image_grid_thw`, or model-specific media tensors;
- optional packing metadata owned by the packing skill.

### 4. Implement Preprocess And Guards

Preprocessing should:

- normalize roles before rendering;
- validate placeholder/media count before expensive media IO;
- use the target processor's real chat template and media token rules;
- build assistant-only labels and mask prompt/media/pad positions;
- drop or sentinel samples with no supervised tokens;
- keep media refs aligned with rendered placeholders.

Use staged guards before preprocessing, after tokenization, after media
materialization, and after optional packing when the backend supports those
boundaries.

### 5. Build Processor And Collator

Processor setup should normalize tokenizer padding, media pixel limits, and any
cache/fingerprint behavior needed by the backend.

The collator should:

- pad `input_ids`, `attention_mask`, and `labels`;
- pad labels with the ignore index;
- concatenate media tensors in placeholder order;
- handle text-only local batches in a model-valid way;
- reject mixed packed/unpacked samples unless the recipe intentionally supports
  mixing.

### 6. Keep Packing Separate

If packing is enabled, this skill owns raw-to-tokenized-to-collated behavior.
`vlm-packing` owns grouping, packed metadata, packed attention masks, and packed
loss/token accounting.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- backend lifecycle and media materialization point are explicit;
- raw schema matches processor placeholder and media-count conventions;
- chat rendering and media token expansion are target-model specific;
- labels supervise only intended assistant tokens;
- text-only, single-image, and multi-image samples remain model-valid;
- invalid samples are filtered, sentineled, or failed consistently;
- collator preserves token/media alignment and label ignore regions;
- packing responsibilities are delegated to `vlm-packing` when enabled;
- no repo-wide data-pipeline behavior was added to `mvp_engine/`;
- structure or smoke checks are not reported as completed sample-matrix impact
  validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/vlm-data-pipeline/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

## Output

- State whether the work modified an existing pipeline or ported a new one.
- State the dataset backend, raw schema, and materialization boundary.
- State the invalid-sample policy and collator behavior.
- State whether packing is involved and what was delegated to `vlm-packing`.
- Report soft validation and hard validation status.
- Call out remaining gaps, such as no sample-matrix impact validation.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/pipeline_rules.md`: backend, schema, preprocessing, guard,
  processor, collator, and sample-matrix rules.
