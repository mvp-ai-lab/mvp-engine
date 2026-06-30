---
name: vlm-packing
description: Add, review, update, and validate VLM/MLLM packing behavior around
  the current MLLMDataKit design, including data_kit.PackingSpec,
  MLLMPackingAssembler, MLLMPack metadata, block-causal masks, model-specific
  packed input preparation, token accounting, and step estimation.
---

# VLM Packing

## Goal

Use `data_kit.PackingSpec` through `MLLMDataKit.build_dataset(data_spec)` for standard
MLLM packing. This skill covers packing knobs and model-specific packed
semantics around the generic DataKit packer.

## Required Inputs

- target recipe path;
- existing `data_kit.DataSpec` and `data_kit.PackingSpec` wiring;
- `max_seq_len`, strategy, open-pack limit, and buffer size;
- packed batch fields consumed by model preparation;
- attention backend and position-id requirements;
- token accounting and total-step estimation path.

Ask only if packed attention semantics or accounting boundaries are unclear.

## Workflow

### 1. Confirm DataKit Packing

Search:

```bash
rg -n \
  "MLLMDataKit|\\.PackingSpec|pack_segment_ids|source_sample_num|prepare_packed" \
  recipes/<recipe> mvp_engine/kit
```

For standard MLLM recipes, packing is part of the DataKit pipeline. Active
configuration lives in `data_kit.PackingSpec`.

### 2. Leave Generic Packing In DataKit

DataKit owns:

- grouping tokenized `MLLMSample` objects;
- producing `MLLMPack`;
- concatenating token fields in `MLLMPack.to_model_inputs()`;
- creating `pack_segment_ids` and `source_sample_num`;
- merging media fields through `MLLMMediaHandler.merge_pack`.

Recipe code consumes these packed fields and adds model-specific preparation.

### 3. Implement Model-Specific Packed Preparation

Recipe/model code may transform packed metadata into:

- block-causal masks;
- packed position ids;
- FlashAttention/cu-seqlens metadata;
- multimodal position rules;
- backend-specific mask patches;
- FLOPs, token, and loss accounting metadata.

Keep this near the model or engine path that prepares model inputs.

## Validation

Soft checks:

- packing knobs map to `data_kit.PackingSpec`;
- custom algorithms use `data_kit.PackingSpec(assembler_cls=...)`;
- packed samples include `pack_segment_ids` and `source_sample_num`;
- collator pads packed metadata with inactive values;
- model preparation prevents cross-sample attention and preserves multimodal
  position behavior;
- total/effective token counts and step estimation count packed outputs
  consistently.

Hard checks when requested:

```bash
.venv/bin/python -m compileall -q mvp_engine/kit/mllm/data recipes/<recipe>
.venv/bin/python -m pytest recipes/<recipe>/tests/test_structure.py -q
.venv/bin/python -m pytest recipes/<recipe>/tests/test_smoke.py -q
```

## Output

- State DataKit packing knobs and any custom assembler.
- State attention, position, and token accounting behavior.
- Report validation and untested packed-boundary risks.

## Read On Demand

- `skills/kit/mllm-data-kit/references/packing.md`: standard DataKit packing contract.
- `references/packing_rules.md`: detailed checks for custom packed model behavior.
