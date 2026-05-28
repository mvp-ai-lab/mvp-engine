---
name: vlm-data-pipeline
description: Add, review, update, and validate VLM data pipelines using the
  MLLM data kits first, including raw schema normalization, media extensions,
  processor setup, always-on packing, materialization, collation, and recipe
  integration.
---

# VLM Data Pipeline

## Goal

Use `MLLMDataKit`, `MLLMSampleKit`, `MLLMMediaKit`, and `PackingOptions` as the
default VLM data implementation. Add recipe-local code only for schema,
modality, model-family, or backend behavior that the standard kits do not cover.

## Required Inputs

- target recipe path and engine data entrypoint;
- raw row schema and media placeholder convention;
- target processor and model-facing batch fields;
- dataset backend and `resolve_ref` lifecycle;
- whether default Qwen-style image media is sufficient;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask only if raw schema, media semantics, or dataset backend cannot be derived.

## Workflow

### 1. Locate Existing Kit Wiring

Search:

```bash
rg -n "MLLMDataKit|MLLMSampleKit|MLLMMediaKit|PackingOptions|build_dataset|build_collator" recipes/<recipe>
```

If the recipe can use the standard pipeline, wire `MLLMDataKit` instead of
copying preprocessing, packing, materialization, or collation logic.

### 2. Choose The Extension Point

- Raw field names, role aliases, placeholder parsing, and media ordering:
  subclass or configure `MLLMSampleKit`.
- Media token counts, placeholder expansion, label masking, image/video/audio
  loading, processor media tensors, and batch media fields: subclass
  `MLLMMediaKit`.
- Dataset backend, guard placement, chat-SFT turn construction, packing
  lifecycle, or dataloader policy: subclass `MLLMDataKit`.

Read `skills/kit/mllm-data-kit/SKILL.md` before implementing any custom data
behavior.

### 3. Preserve The Standard Lifecycle

The normal flow is:

```text
raw row -> SampleKit.normalize -> MediaKit.prepare/render_text
-> tokenize/mask labels -> pack -> resolve_ref -> MediaKit.materialize
-> finalize packed media -> collate -> model batch
```

Heavy media IO, such as video frame sampling, should happen in
`MLLMMediaKit.materialize()` after refs are resolved.

### 4. Keep Packing Integrated

The standard MLLM DataKit pipeline is always packed. Do not add a `data.packing`
boolean for OpenBee-style recipes. Expose only active `PackingOptions` knobs.

Use `skills/data/vlm-packing/SKILL.md` only for model-specific packed attention,
position ids, or accounting behavior outside DataKit's generic packing.

## Validation

### Soft Validation

- standard data flow uses kit APIs instead of duplicated recipe-local logic;
- custom SampleKit, MediaKit, or DataKit code has a clear boundary;
- media refs stay aligned with placeholders through packing and materialization;
- labels supervise only intended assistant tokens;
- text-only, single-media, multi-media, and invalid samples have explicit
  behavior;
- no stale `data.packing` boolean is introduced for standard MLLM recipes.

### Hard Validation

Run:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

Add recipe-local skill assertions only when a custom schema or modality contract
is not covered by existing tests.

## Output

- State which kit APIs were used or extended.
- State raw schema, media lifecycle, packing knobs, and collator outputs.
- Report validation and remaining untested modality cases.

## Read On Demand

- `skills/kit/mllm-data-kit/SKILL.md`: authoritative kit API and extension
  guide.
- `references/pipeline_rules.md`: legacy backend and sample-matrix checks.
