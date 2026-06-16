---
name: vlm-freeze-policy
description: Add, review, update, and validate VLM freeze policy using
  MLLMModelKit when possible, including vision/projector/language prefix groups,
  optimizer interaction, trainable dtype handling, and MFU impacts.
---

# VLM Freeze Policy

## Goal

Use `MLLMModelKit.apply_freeze_policy()` for standard VLM trainability control.
Add recipe-local freeze logic only when prefix groups are not expressive enough
for the target model.

## Required Inputs

- target recipe path and `prepare_model()` / optimizer paths;
- freeze config fields and stage defaults;
- real `named_parameters()` output;
- model component boundaries: vision, projector/connector, language model;
- optional FLOPs/MFU dependency on freeze state.

## Workflow

### 1. Prefer MLLMModelKit

```python
model = self.model_kit.apply_freeze_policy(
    model,
    freeze_vit=self.config.model.freeze_vit,
    freeze_projector=self.config.model.freeze_projector,
    freeze_llm=self.config.model.freeze_llm,
)
```

For non-Qwen names, pass explicit `vit_prefixes`, `projector_prefixes`, and
`llm_prefixes`.

### 2. Preserve Build Order

Use this order unless the recipe documents a concrete exception:

```text
load model -> recipe patches -> token-loss/FLOPs patches -> freeze policy
-> trainable dtype upcast/counts -> checkpointing -> compile -> parallelize
-> build optimizer
```

### 3. Fall Back Only For Non-Prefix Policies

Use recipe-local logic only for policies based on module type, regex, partial
layer ranges, or dynamic trainability. Keep the result visible through
`parameter.requires_grad` before optimizer construction.

## Validation

### Soft Validation

- freeze goes through `MLLMModelKit` or the fallback is justified;
- prefixes match real parameter names and groups are non-overlapping;
- at least one parameter remains trainable for each intended stage;
- optimizer, trainable dtype upcast, logs, and MFU use `requires_grad`;
- freeze happens before distributed wrapping and optimizer construction.

### Hard Validation

Run structure and smoke tests for every changed stage or representative config.

## Output

- State freeze flags and component prefixes.
- State build-order placement and optimizer/MFU impacts.
- Report validation and remaining runtime gaps.

## Read On Demand

- `skills/kit/mllm-model-kit/references/freeze-policy.md`.
- `references/patterns.md`: legacy grouping and FLOPs notes.
