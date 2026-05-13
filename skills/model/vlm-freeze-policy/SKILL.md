---
name: vlm-freeze-policy
description: Add or adapt a recipe-local VLM freeze policy with independently configurable trainable groups such as vision encoder, multimodal projector, and language model, including config and optional FLOPs/MFU wiring.
---

# VLM Freeze Policy

## Goal

Make VLM submodules independently frozen or trainable from recipe config. Keep the implementation recipe-local unless the user explicitly asks for engine-wide behavior.

Typical groups are:

- vision encoder: patch embedding, vision tower, or ViT blocks
- connector: projector, merger, adapter, resampler, or cross-modal bridge
- language stack: text backbone and output head

Use the real model's parameter names. Do not assume every VLM uses the same prefixes.

## Workflow

### 1. Inspect The Current Recipe

- Find the model builder and the point where optimizer parameters are collected.
- Find config schema and YAML / launch config files.
- Find any code that depends on `requires_grad`, such as fp32 trainable-parameter upcasting, parameter counting, optimizer filtering, FSDP/DDP wrapping, and FLOPs/MFU estimation.
- Search existing code for broad freezes such as `model.visual`, `vision_tower`, `requires_grad = False`, or existing freeze flags.

### 2. Derive Logical Parameter Groups

Use `model.named_parameters()` naming conventions from the actual model. Define non-overlapping groups that match the training stages the recipe needs.

Guidelines:

- Keep the connector/projector separate from the vision encoder so alignment stages can train only the connector.
- Keep output heads with the language stack unless the recipe has a reason to control them separately.
- Prefer prefixes or module-path predicates over fragile substring checks.
- If one parameter could match multiple groups, define a deterministic priority and count it once.

Example shape:

```python
VISION_PREFIXES = (...)
CONNECTOR_PREFIXES = (...)
LANGUAGE_PREFIXES = (...)


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)
```

### 3. Implement A Recipe-Local Freeze Helper

Add a small helper in the recipe model module. Match the local style: explicit boolean arguments are fine for a small fixed set of groups; a mapping is better when the model has many groups.

Explicit-flag shape:

```python
def apply_freeze_policy(
    model,
    *,
    freeze_vision: bool = False,
    freeze_connector: bool = False,
    freeze_language: bool = False,
) -> dict[str, int]:
    frozen_counts = {"vision": 0, "connector": 0, "language": 0}

    for name, parameter in model.named_parameters():
        if freeze_vision and _matches(name, VISION_PREFIXES):
            parameter.requires_grad = False
            frozen_counts["vision"] += parameter.numel()
        elif freeze_connector and _matches(name, CONNECTOR_PREFIXES):
            parameter.requires_grad = False
            frozen_counts["connector"] += parameter.numel()
        elif freeze_language and _matches(name, LANGUAGE_PREFIXES):
            parameter.requires_grad = False
            frozen_counts["language"] += parameter.numel()

    return frozen_counts
```

Mapping shape:

```python
def apply_freeze_policy(model, freeze_policy: dict[str, bool]) -> dict[str, int]:
    frozen_counts = {group: 0 for group in freeze_policy}
    ...
```

Do not add new dependencies for this.

### 4. Wire The Build Order

Call the freeze helper after the model has all trainable modules attached or replaced, and before any step that consumes `requires_grad`.

Common ordering:

1. load model
2. apply model patches, adapters, forward injections, and checkpointing hooks
3. apply freeze policy
4. upcast or count trainable parameters
5. parallelize and build optimizer

If the recipe exports model helpers, export the freeze helper only when other code or tests need it.

### 5. Add Config Controls

Add config fields near related model-loading or training-stage options. Preserve the recipe's existing schema strictness.

Choose defaults from the recipe's intended default training stage, not from a hard-coded global rule. If stage files already express different phases, make each stage explicit in YAML so launch behavior is obvious.

Example YAML shape:

```yaml
model:
  freeze_vision: true
  freeze_connector: false
  freeze_language: true
```

Use existing names when the recipe already has conventions such as `freeze_vit`, `freeze_merger`, or `freeze_llm`.

### 6. Update Optimizer And Metrics Wiring

- If the optimizer already filters `parameter.requires_grad`, confirm the freeze helper runs before optimizer construction.
- If the optimizer currently receives all parameters, change it to trainable parameters only when that matches the recipe style.
- If logs include trainable parameter counts, make sure they observe the post-freeze model.

### 7. Account For FLOPs/MFU When Present

If the recipe estimates training FLOPs by component, make the estimate freeze-aware. A fully trainable component usually uses forward + backward weight/activation cost; a frozen component may still need activation-gradient cost if trainable upstream modules depend on it.

For a serial VLM path, reason from loss backward toward inputs:

- language stack receives loss gradients first
- connector receives gradients from the language stack
- vision encoder receives gradients from the connector
- image tensors usually do not require gradients

Thread freeze flags only to the site that actually computes component FLOPs. Avoid passing the same flags into a later logging helper when a precomputed `model_flops_per_step` already includes the freeze-aware result.

If the existing FLOPs path is a coarse total estimate and adding accurate freeze multipliers would be speculative, leave it unchanged and document that MFU remains an all-trainable approximation.

## Validation

- Search for stale broad freezes that prevent independent group control.
- Confirm the model builder applies freezing before trainable-parameter upcasting, parameter counting, optimizer construction, and parallel wrapping.
- Confirm every config / YAML flag is consumed by the model builder.
- Confirm frozen parameter counts are counted once per group.
- Confirm the intended default stage has at least one trainable parameter.


## Read On Demand

- Reference implementation: `references/openbee-freeze-policy.patch` contains a concrete OpenBee recipe adaptation generated from this skill.