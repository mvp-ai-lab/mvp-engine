---
name: gradient-checkpointing
description: Add, review, update, and validate gradient checkpointing through
  MLLMModelKit when possible, with recipe-local fallback only for model-specific
  activation-checkpoint wiring.
---

# Gradient Checkpointing

## Goal

Enable checkpointing without changing model math. For MLLM recipes, use
`MLLMModelKit.apply_gradient_checkpointing()` first. Add manual recipe/model
checkpoint code only when the kit path cannot express the model's repeated
block structure.

## Required Inputs

- target recipe path and `prepare_model()` path;
- config fields for `model.gradient_checkpointing.enabled` and
  `use_reentrant`;
- whether the loaded model supports HF `gradient_checkpointing_enable(...)`;
- distributed wrapping point;
- optional custom target modules.

## Workflow

### 1. Prefer MLLMModelKit

```python
if self.config.model.gradient_checkpointing.enabled:
    model = self.model_kit.apply_gradient_checkpointing(
        model,
        use_reentrant=self.config.model.gradient_checkpointing.use_reentrant,
    )
```

Place this after model construction/patches and before distributed wrapping.

### 2. Use Custom Mode Only When Needed

Use `mode="custom"` or `"hf_with_custom"` with `target_modules` only when HF
native support is absent or incomplete. Keep target names explicit and tied to
runtime module classes.

### 3. Fall Back To Recipe-Local Manual Logic Last

Manual checkpointing should live in the recipe/model implementation and only
wrap repeated blocks. Preserve parameter names, checkpoint keys, forward outputs,
and loss math.

Read `skills/kit/mllm-model-kit/SKILL.md` before editing.

## Validation

### Soft Validation

- checkpointing is enabled through `MLLMModelKit` unless a clear fallback is
  documented;
- config defaults and stage overrides are intentional;
- cache/attention outputs do not conflict with checkpointing;
- checkpointing runs before FSDP/DDP/TP wrapping;
- manual wrapping, if any, is limited to repeated blocks.

### Hard Validation

Run structure and smoke tests. When possible, run smoke with:

```bash
--config-override model.gradient_checkpointing.enabled=true
```

## Output

- State whether kit native, kit custom, or recipe-local fallback was used.
- State where checkpointing runs relative to patches and distributed wrapping.
- Report validation and runtime gaps.

## Read On Demand

- `skills/kit/mllm-model-kit/references/gradient-checkpointing.md`.
- `references/patterns.md`: legacy manual adaptation examples.
