---
name: vlm-data-pipeline
description: Add, review, update, and validate VLM/MLLM data pipelines using
  the current spec/handler-based MLLMDataKit design. Use for raw schema
  normalization, data_kit.DataSpec wiring, source resample/resolve_refs policy, media
  extensions, processor setup, packing, guards, collation, step estimation, and
  recipe integration.
---

# VLM Data Pipeline

## Goal

Use `MLLMDataKit` and explicit `data_kit.DataSpec` objects as the default VLM data
implementation. Recipe-local code should describe recipe-specific schema,
modality, tokenization, packing, model-input preparation, or backend lifecycle
behavior.

Read `skills/kit/mllm-data-kit/SKILL.md` before implementing custom MLLM data
behavior. Treat that skill as the authoritative API contract.

## Required Inputs

- target recipe path and engine data entrypoint;
- raw row schema and media placeholder convention;
- target processor and model-facing batch fields;
- dataset backend and whether refs should be resolved;
- training versus estimation source policy: `resample` and `resolve_refs`;
- packing knobs and packed model-input preparation path;
- recipe-local structure/smoke tests.

Ask only if raw schema, media semantics, or dataset backend cannot be derived
from the repository.

## Workflow

### 1. Locate Existing Kit Wiring

Search:

```bash
rg -n \
  "MLLMDataKit|\\.DataSpec|\\.SampleSpec|\\.SourceSpec|\\.PackingSpec|QwenVLChatSchemaHandler|build_dataset" \
  recipes/<recipe>
```

For standard MLLM recipes, build specs directly in the engine. Express
preprocessing, packing, media materialization, and collation differences through
handlers or specs.

### 2. Choose The Extension Point

- Source path, backend, sharding, `resample`, or `resolve_refs`:
  `data_kit.SourceSpec` / `data_kit.DistributionSpec`.
- Raw row format, role aliases, prompt/target split, placeholders, media slot
  binding, or label policy: `MLLMSchemaHandler`.
- Placeholder rendering, image/video/audio decode, model media tensors, pack
  merge, or batch collation: `MLLMMediaTypeHandler` registered in
  `MLLMMediaHandler`.
- Tokenization, truncation, or ignore-index behavior:
  `MLLMTokenizationHandler`.
- Packing strategy, buffer, open-pack limit, or custom packer:
  `data_kit.PackingSpec`.
- Dataloader batch shape and worker settings: `data_kit.LoaderSpec`.
- Model-specific packed attention, position ids, FlashAttention metadata, or
  dummy media paths: recipe/model preparation code outside generic DataKit.
- Dataset stage order or backend lifecycle: extend `MLLMDataKit`.

### 3. Preserve The Standard Lifecycle

The current flow is:

```text
source -> raw guard -> MLLMSample -> sample guard -> packing
-> optional resolve_ref -> MLLMPack.to_model_inputs -> model-input guard
-> MLLMBatchCollator -> optional recipe/model batch preparation
```

Keep heavy media IO after reference resolution. Keep label policy in schema
segments. Keep model-specific attention and position semantics outside the
generic data kit.

### 4. Keep Train And Estimation Specs Explicit

Training source spec normally uses:

```python
data_kit.SourceSpec(..., resample=True, resolve_refs=True)
```

Step-estimation source spec normally uses:

```python
data_kit.SourceSpec(..., resample=False, resolve_refs=False)
```

Pass the finite packed estimation dataset to `MLLMStepEstimationKit`. Keep these
choices visible on the source spec used at each engine callsite.

## Validation

Soft checks:

- engine builds `data_kit.DataSpec` from explicit source/sample/packing/loader specs;
- media refs stay aligned with placeholders through packing and loading;
- labels supervise only intended segments;
- text-only, single-media, multi-media, invalid rows, and unreadable media have
  explicit behavior;
- packed model-input preparation preserves segment isolation.

Hard checks when requested:

```bash
.venv/bin/python -m compileall -q mvp_engine/kit/mllm/data recipes/<recipe>
.venv/bin/python -m pytest recipes/<recipe>/tests/test_structure.py -q
.venv/bin/python -m pytest recipes/<recipe>/tests/test_smoke.py -q
```

## Output

- State which specs and handlers were used or extended.
- State source schema, media lifecycle, packing knobs, and collator outputs.
- State model-specific packed input preparation.
- Report validation and remaining untested modality cases.

## Read On Demand

- `skills/kit/mllm-data-kit/SKILL.md`: authoritative kit API and extension guide.
- `references/pipeline_rules.md`: detailed review checklist for custom pipelines.
