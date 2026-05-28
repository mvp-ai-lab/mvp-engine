---
name: vlm-packing
description: Add, review, update, and validate VLM packing behavior around the
  MLLM data kits, including PackingOptions, packed metadata, model input
  preparation, attention isolation, token accounting, and custom fallback paths.
---

# VLM Packing

## Goal

Use `MLLMDataKit.build_dataset(..., packing=PackingOptions(...))` for standard
sample packing. This skill is for packing knobs and model-specific packed
semantics, not for reimplementing the generic packer.

## Required Inputs

- target recipe path;
- existing `MLLMDataKit` and `PackingOptions` wiring;
- `max_seq_len`, packing strategy, open-pack limit, and buffer size;
- packed batch fields consumed by model preparation;
- attention backend and position-id requirements;
- token accounting and step-inference paths.

Ask only if packed attention semantics or accounting boundaries are unclear.

## Workflow

### 1. Confirm DataKit Packing

Search:

```bash
rg -n "PackingOptions|build_packing_assembler|pack_segment_ids|source_sample_num|prepare_packed" recipes/<recipe> mvp_engine/kit
```

For standard MLLM recipes, packing is always enabled. Do not add a config field
that disables it. Keep config to strategy/buffer/open-pack knobs.

### 2. Leave Generic Packing In The Kit

DataKit owns:

- grouping tokenized samples;
- concatenating `input_ids`, `attention_mask`, and `labels`;
- creating `pack_segment_ids` and `source_sample_num`;
- merging media refs/tensors through MediaKit finalization hooks.

Do not duplicate these in the recipe unless the recipe intentionally avoids
`MLLMDataKit`.

### 3. Implement Model-Specific Packed Preparation

Recipe/model code may still need to transform packed metadata into:

- block causal masks;
- packed position ids;
- FlashAttention/cu-seqlens metadata;
- multimodal position rules;
- packed FLOPs or token accounting metadata.

Keep this near the model or engine path that prepares model inputs.

## Validation

### Soft Validation

- packing knobs map to `PackingOptions`;
- packed samples include `pack_segment_ids` and `source_sample_num`;
- collator pads packed metadata with inactive values;
- model preparation prevents cross-sample attention and preserves multimodal
  position behavior;
- total/effective token counts and step inference count packed outputs
  consistently;
- no standard MLLM config adds `data.packing`.

### Hard Validation

Run:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

Add impact validation only when attention isolation or packed position behavior
cannot be verified by smoke tests.

## Output

- State DataKit packing knobs and model-specific packed preparation.
- State attention/position/token accounting behavior.
- Report validation and any untested packed-boundary risks.

## Read On Demand

- `skills/kit/mllm-data-kit/references/packing.md`: standard DataKit packing
  contract.
- `references/packing_rules.md`: detailed legacy checks for custom packers.
