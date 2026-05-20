---
name: vlm-freeze-policy
description: Add, review, update, and validate recipe-local VLM freeze policy for independently configurable vision encoder, connector/projector, and language model parameter groups, including optimizer and optional FLOPs/MFU integration.
---

# VLM Freeze Policy

## Goal

Add or maintain recipe-local VLM freeze policy without changing model math:

- expose independent trainability controls for vision, connector, and language
  components;
- derive parameter groups from the real model's `named_parameters()`;
- apply freezing before optimizer construction and distributed wrapping;
- keep optimizer, trainable-parameter logs, precision upcasting, and optional
  FLOPs/MFU paths consistent with `requires_grad`.

## Required Inputs

Identify these before editing:

- target recipe path;
- config schema and YAML configs;
- model builder and engine `prepare_model()` / `prepare_optimizer()` paths;
- loaded model class and real parameter names;
- VLM component boundaries: vision encoder, connector/projector/resampler, and
  language model;
- optional FLOPs/MFU implementation if the recipe reports training FLOPs;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if component ownership or intended stage defaults cannot be
derived from the model and recipe configs.

## Workflow

### 1. Locate Runtime Integration Points

Search the recipe first:

```bash
rg -n "freeze_|requires_grad|named_parameters|apply_freeze|trainable|calculate_model_flops|mfu" recipes/<recipe>
```

Find:

- where the model is loaded and patched;
- where the optimizer collects parameters;
- where trainable parameter counts or dtype upcasting happen;
- where distributed wrapping happens;
- whether FLOPs/MFU depends on trainability.

### 2. Define Parameter Groups

Use the real loaded model's `named_parameters()` output. Do not assume Qwen,
LLaVA, CLIP, or HF naming conventions.

Typical groups:

- vision encoder: patch embedding, vision tower, ViT blocks, visual trunk;
- connector: projector, merger, adapter, resampler, Q-former, cross-modal bridge;
- language model: text backbone, decoder blocks, embeddings, and output head.

Prefer explicit prefixes or module-path predicates. Keep groups deterministic,
non-overlapping, and easy to audit. Read `references/patterns.md` for naming and
grouping examples.

### 3. Add Config

Expose freeze controls under `model`. Preserve existing field names when the
recipe already has them:

```yaml
model:
  freeze_vit: true
  freeze_merger: false
  freeze_llm: true
```

For new recipes, choose names that match the recipe's model vocabulary, such as
`freeze_vision_encoder`, `freeze_projector`, and `freeze_language_model`.

Defaults should describe the recipe's intended default training stage. Keep
stage YAMLs explicit when different stages train different components.

### 4. Apply Freeze Policy

Apply freezing after the model has all trainable modules attached or patched,
and before anything consumes `requires_grad`:

```text
1. load model
2. apply recipe model patches, checkpointing hooks, and forward injections
3. apply freeze policy
4. upcast or count trainable parameters
5. compile if enabled
6. parallelize model
7. build optimizer from trainable parameters
```

The freeze helper should set `parameter.requires_grad = False` only for matched
frozen groups. It should leave unmatched parameters unchanged or fail loudly if
the recipe requires complete coverage.

### 5. Update Dependent Paths

Check every path that depends on trainability:

- optimizer parameter collection must use `parameter.requires_grad`;
- trainable parameter counts and logs must reflect the freeze policy;
- fp32 trainable-parameter upcasting must skip frozen parameters;
- distributed wrapping must not happen before freeze state is set;
- FLOPs/MFU, if present, must receive freeze state where the estimate is
  computed.

For FLOPs, do not assume frozen means forward-only. A frozen module can still
need input-gradient compute if gradients flow through it into trainable upstream
or downstream modules. Use the MFU skill's FLOPs reference when the recipe
reports MFU.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- parameter group predicates match real model parameter names;
- groups are non-overlapping and cover the intended VLM components;
- config exposes the intended freeze flags and stage YAMLs set them explicitly
  when stages differ;
- every freeze flag is consumed by the model builder or freeze helper;
- freeze policy runs before optimizer construction and distributed wrapping;
- optimizer, trainable-parameter logging, and trainable dtype upcasting use
  `requires_grad`;
- the intended stage has at least one trainable parameter;
- optional FLOPs/MFU logic accounts for freeze state without treating every
  frozen module as automatically forward-only;
- no repo-wide freeze wrapper was added to `mvp_engine/`;
- CPU-only or structure-only checks are not reported as completed runtime
  freeze-policy validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/vlm-freeze-policy/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

If the recipe has multiple freeze stages, run smoke validation for the stage or
config overrides changed by the task.

## Output

- State which component groups and config flags were used.
- State where freeze policy is applied relative to model patches, compile,
  distributed wrapping, and optimizer construction.
- State optimizer, trainable-parameter logging, and optional FLOPs/MFU impacts.
- Report soft validation and hard validation status.
- Call out any remaining gap, such as no GPU/NPU environment for runtime smoke.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/patterns.md`: parameter-group, build-order, and FLOPs/MFU patterns.
