---
name: mllm-model-kit
description: Use and extend MLLMModelKit for multimodal model loading, recipe
  patches, freeze policy, gradient checkpointing, and torch.compile wiring in
  MVP-Engine recipes.
---

# MLLM Model Kit

## Goal

Use `MLLMModelKit` as the default model setup API for MLLM recipes:

- `build_model(...)` loads HF image-text models;
- `apply_model_patches(...)` applies recipe-specific runtime patches;
- `apply_freeze_policy(...)` freezes vision/projector/language prefix groups;
- `apply_gradient_checkpointing(...)` enables HF or custom activation
  checkpointing;
- `apply_model_compile(...)` applies `model.compile(...)` with the kit's
  standard VLM graph-break handling.

## Required Inputs

Identify these before editing:

- target recipe engine and `prepare_model()` path;
- config fields for pretrained model, dtype, attention backend, freeze flags,
  checkpointing, compile, and recipe patches;
- real parameter names for freeze prefixes;
- whether the model supports HF gradient checkpointing;
- whether compile should cover the full model or a recipe-specific submodule.

Ask the user only when component ownership, compile target, or checkpointing
mode cannot be derived.

## Workflow

### 1. Use The Kit In Prepare Model

Prefer this order:

```python
from mvp_engine.kit import MLLMModelKit

self.model_kit = MLLMModelKit()
model = self.model_kit.build_model(...)
model = self.model_kit.apply_model_patches(model, [...])
model = self.model_kit.apply_freeze_policy(model, freeze_vit=..., freeze_projector=..., freeze_llm=...)
if config.model.gradient_checkpointing.enabled:
    model = self.model_kit.apply_gradient_checkpointing(model, use_reentrant=...)
if config.model.compile.enabled:
    model = self.model_kit.apply_model_compile(model, backend=..., mode=...)
model = parallelize_model(model, ...)
```

Apply freeze before optimizer construction and distributed wrapping. Apply
compile before distributed wrapping unless a recipe documents a concrete
exception.

### 2. Override Only The Unstable Boundary

- Use `apply_model_patches()` for model-family patches such as FLOPs injection
  or backend compatibility patches.
- Pass custom freeze prefixes to `apply_freeze_policy()` when default Qwen-style
  prefixes do not match the target model.
- Use `mode="custom"` or `mode="hf_with_custom"` for checkpoint wrapping only
  when HF native checkpointing is insufficient.
- Add recipe-local compile logic only when the kit's full-model compile scope is
  wrong for that model.

Read the focused references before changing a feature-specific skill or recipe
implementation.

## Validation

### Soft Validation

- model construction still happens once on the real training path;
- recipe patches run before features that depend on patched methods;
- freeze policy runs before trainable dtype upcast, optimizer construction, and
  distributed wrapping;
- checkpointing disables cache when required and runs before distributed
  wrapping;
- compile scope is intentional and does not change forward outputs, parameter
  names, or checkpoint keys;
- feature skills call the kit first and only describe recipe-local fallback when
  the kit API does not fit.

### Hard Validation

Run the recipe's normal structure and smoke tests. For compile/checkpointing,
run smoke with the relevant config enabled when the environment supports it.

## Output

- State which `MLLMModelKit` methods are used.
- State recipe-local patches and any custom prefixes or checkpoint targets.
- State feature ordering relative to parallelization and optimizer creation.
- Report validation commands and residual runtime gaps.

## Read On Demand

- `references/freeze-policy.md`: freeze prefix and trainability rules.
- `references/gradient-checkpointing.md`: native and custom checkpointing rules.
- `references/model-compile.md`: compile scope and placement rules.
