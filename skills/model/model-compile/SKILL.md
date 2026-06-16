---
name: model-compile
description: Add, review, update, and validate torch model.compile support using
  MLLMModelKit when possible, including compile config, scope, placement before
  distributed wrapping, and recipe-local fallback for special graph boundaries.
---

# Model Compile

## Goal

Wire `torch.compile` without changing model math. For MLLM recipes, use
`MLLMModelKit.apply_model_compile()` first. Add recipe-local compile code only
when the safe compile target is not the kit's standard target.

## Required Inputs

- target recipe path and `prepare_model()` path;
- config fields for `model.compile.enabled`, backend, and mode;
- compile target and known graph-break regions;
- distributed wrapping point;
- runtime validation environment.

## Workflow

### 1. Prefer MLLMModelKit

```python
if self.config.model.compile.enabled:
    model = self.model_kit.apply_model_compile(
        model,
        backend=self.config.model.compile.backend,
        mode=self.config.model.compile.mode,
    )
```

Run after model construction, recipe patches, freeze policy, and checkpointing;
run before distributed wrapping unless a recipe documents an exception.

### 2. Fall Back For Special Compile Scope

Use recipe-local compile code only when:

- only a submodule should be compiled;
- the model has non-Qwen graph-break handling;
- distributed wrappers require a documented different placement.

Keep checkpoint keys, parameter names, forward outputs, and loss math unchanged.

## Validation

### Soft Validation

- compile goes through `MLLMModelKit` unless fallback is documented;
- config defaults and enabled stages are intentional;
- compile scope matches the real training hot path;
- compile placement relative to patches, checkpointing, freeze, and distributed
  wrapping is correct;
- CPU-only checks are not reported as runtime compile validation.

### Hard Validation

Run structure and smoke tests. When possible, run smoke with compile enabled:

```bash
--config-override model.compile.enabled=true
```

## Output

- State kit or fallback compile path.
- State compiled target and placement.
- Report validation and any performance/runtime gap.

## Read On Demand

- `skills/kit/mllm-model-kit/references/model-compile.md`.
- `references/patterns.md`: legacy compile-scope examples.
