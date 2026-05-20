---
name: vlm-packing
description: Add, review, update, and validate recipe-local VLM sample packing,
  including grouping, packed metadata, collator padding, packed model-input
  preparation, attention isolation, multimodal alignment, and packed training
  accounting.
---

# VLM Packing

## Goal

Add or modify VLM sample packing without changing model semantics:

- group already-tokenized samples up to `max_seq_len`;
- concatenate token fields while preserving source-sample boundaries;
- merge image/video payloads in placeholder order;
- create boundary metadata for attention, position ids, or backend-specific
  sequence metadata;
- prevent cross-sample attention, position, and loss leakage;
- keep token accounting and step inference aligned with packed outputs;
- keep packing recipe-local unless the user explicitly asks for shared engine
  behavior.

Use `recipes/basic_vlm` as the reference implementation, not as a required
shape for every backend.

## Required Inputs

Identify these before editing:

- target recipe path;
- dataset backend and packing lifecycle;
- tokenized sample contract from the VLM data pipeline;
- `max_seq_len`, packing strategy, buffer, and open-pack knobs;
- media payload fields and media grid metadata;
- model-specific packed attention and position-id requirements;
- training accounting path for tokens, steps, throughput, and MFU;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if packing point, model attention backend, or accounting
boundary cannot be derived from the recipe.

## Workflow

### 1. Locate Existing Packing

Search the recipe first:

```bash
rg -n "packing|pack_segment|source_sample|prepare_packed|position_ids" recipes/<recipe>
rg -n "attention_mask|image_grid_thw|pixel_values|effective_token|total_token" recipes/<recipe>
```

Find:

- config knobs;
- dataset grouping/finalization point;
- collator metadata padding;
- model-input preparation before forward;
- packed step/token accounting.

### 2. Identify Backend And Lifecycle

Packing should normally happen after tokenization and before expensive media
materialization when media loading is costly.

Read `references/packing_rules.md` when backend lifecycle, materialization
point, seed source, or `drop_last` semantics are unclear.

### 3. Add Or Update Packing Config

Expose only knobs used by active code:

```python
packing: bool = False
packing_selection_strategy: Literal["random", "best_fit"] = "best_fit"
packing_open_pack_limit: int = Field(8, ge=1)
packing_buffer_size: int = Field(64, ge=0)
```

Keep defaults conservative unless active recipe configs intentionally enable
packing.

### 4. Implement Packer And Finalizer

The packer groups valid tokenized samples. The finalizer creates one
model-facing packed sample.

It must:

- keep samples in deterministic source order inside each packed output;
- never exceed `max_seq_len` except for explicitly allowed standalone
  overlength samples;
- concatenate `input_ids`, `attention_mask`, and `labels`;
- build segment/boundary metadata;
- merge media tensors or refs in placeholder order;
- record `source_sample_num` or equivalent packed-output accounting metadata.

### 5. Update Collator And Model Preparation

The collator should pad packed metadata and reject mixed packed/unpacked batches
unless explicitly supported.

Model preparation should convert packed metadata into the target model's
expected form: block causal masks, packed position ids, cu-seqlens, or
backend-specific attention metadata.

If the model uses FlashAttention, SDPA, or custom multimodal position logic,
verify that the packed metadata survives every path used in training.

### 6. Update Accounting

Packing changes the unit consumed by training. Recheck:

- total-step inference;
- gradient accumulation;
- total and effective token counts;
- loss normalization;
- throughput and MFU logs;
- late media failures after packing.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- packing is applied after tokenization and before/after materialization by
  explicit design;
- groups respect `max_seq_len`, strategy, buffer, and finish/drop policy;
- labels, masks, media payloads, and boundary metadata stay aligned with
  `input_ids`;
- collator pads metadata with inactive values and rejects unsupported mixing;
- model preparation blocks cross-source attention and preserves multimodal
  position rules;
- packed token accounting and step inference count packed outputs consistently;
- packing responsibilities stay recipe-local;
- structure or smoke checks are not reported as completed packing-impact
  validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/vlm-packing/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q --config-override data.packing=true
```

Add optional impact validation when the task requires proof of packed boundary
behavior. Use recipe-local files such as:

```text
recipes/<recipe>/tests/skills/vlm-packing/test_attention_isolation_impact.py
```

The attention isolation test should prove packed segments cannot
attend across source-sample boundaries.

## Output

- State which packing layer changed: config, dataset, packer, collator, model
  preparation, or accounting.
- State packing point, finalization point, and boundary metadata.
- State model-specific packed attention or position behavior.
- Report soft validation and hard validation status.
- Call out remaining gaps, such as no packing matrix or attention isolation
  impact validation.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/packing_rules.md`: packing lifecycle, metadata, model preparation,
  accounting, and impact validation rules.
