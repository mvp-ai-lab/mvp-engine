---
name: gradient-checkpointing
description: Add gradient checkpointing to a recipe in this repo. Use when checking whether a model already supports checkpointing, wiring recipe config and engine toggles, or adding minimal model-side checkpoint logic and tests.
---

# Gradient Checkpointing

## Goal

- Enable gradient checkpointing on the target recipe without changing model math.
- Keep the implementation recipe-local instead of introducing a repo-wide wrapper.
- Add config, engine wiring, and tests that prove the feature is actually active.

## Required Inputs

- The target recipe path and the files that build the model and engine.
- The top-level model class or the module that owns the repeated layer loop.
- Whether the model already exposes built-in checkpointing support.
- The target recipe's config or schema files.
- A place to add recipe-local tests.

## Workflow

### 1. Classify the model before editing

- Prefer the existing-support path whenever the model already knows how to checkpoint its repeated blocks.
- Use the manual-adaptation path only when checkpointing is not already wired into the model internals.

The existing-support path applies when all of the following are true:
- the top-level model exposes `gradient_checkpointing_enable()` and `gradient_checkpointing_disable()`
- the model propagates `gradient_checkpointing` and `_gradient_checkpointing_func` to the modules that need them
- the repeated blocks already route through `_gradient_checkpointing_func` when `self.gradient_checkpointing and self.training` is true

### 2. Existing-support path: wire the recipe only

- Do not rewrite model internals if the model already supports checkpointing.
- In `prepare_model()`, enable checkpointing after building the model and before FSDP, DDP, or TP wrapping:

```python
gc_enabled = self.config.model.gradient_checkpointing.enabled
gc_use_reentrant = self.config.model.gradient_checkpointing.use_reentrant
if gc_enabled:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc_use_reentrant}
    )
```

- Add config:

```yaml
model:
  gradient_checkpointing:
    enabled: false
    use_reentrant: false
```

- Under the new config system, add `model.gradient_checkpointing` to the recipe schema or `ConfigClass` and read it through typed attribute access in the engine.
- Prefer `use_reentrant: false` unless the target model specifically requires reentrant checkpointing.

### 3. Manual-adaptation path: patch the module that owns the layer loop

- On the module that owns the repeated-layer loop, add:

```python
self.gradient_checkpointing = False
self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
```

- In the loop, call each layer through `_gradient_checkpointing_func` when training with checkpointing enabled.
- Pass only gradient-carrying tensors as explicit checkpoint arguments. Capture masks, RoPE inputs, and other non-differentiable values in a closure.
- If checkpointed layers cannot safely return auxiliary outputs such as attentions or caches, gate checkpointing on those flags or return a consistent reduced output.
- For `PreTrainedModel` subclasses, set `supports_gradient_checkpointing = True`.
- For plain `nn.Module`, implement `gradient_checkpointing_enable()` and `gradient_checkpointing_disable()` locally.

Example:

```python
use_gc = self.gradient_checkpointing and self.training and not output_attentions

for layer in self.layers:
    if use_gc:
        def custom_forward(hidden_states):
            return layer(hidden_states, attention_mask=attention_mask, ...)[0]

        hidden_states = self._gradient_checkpointing_func(custom_forward, hidden_states)
    else:
        hidden_states = layer(hidden_states, attention_mask=attention_mask, ...)[0]
```

### 4. Add recipe-local tests

Add recipe-local tests that cover at least:
- enable and disable toggles set the expected module state
- the checkpoint function is actually invoked during training
- gradients match with and without checkpointing

### 5. Validate the final integration

- Confirm checkpointing is enabled before distributed wrapping.
- Confirm config, engine wiring, and tests all agree on the same feature shape.
- If the model already inherits `GradientCheckpointingLayer` or an equivalent mechanism, do not manually rewrap those blocks.

## Validation

- The chosen path matches the real model capabilities.
- The recipe config exposes `model.gradient_checkpointing.enabled` and `use_reentrant`.
- Checkpointing is enabled before FSDP, DDP, or TP wrapping.
- Recipe-local tests cover toggles, invocation, and gradient consistency.
- The implementation does not introduce a repo-wide wrapper or pass non-differentiable inputs as explicit checkpoint arguments.

Add recipe-local assertions under `recipes/<recipe>/skill_tests/gradient-checkpointing/asserts.py`,
using the standard `assert_structure(...)` and `assert_smoke(...)` hooks:

- `skill_tests/test_structure.py`: verify recipe structure and checkpointing wiring.
- `skill_tests/test_smoke.py`: run one real recipe-owned training step and checkpoint/log path.

## Output

- State which path was used: existing support or manual adaptation.
- State which model, engine, config, and test files were updated.
- Summarize how checkpointing is enabled at runtime.
- Summarize what validation ran and what remains unverified.

## Read On Demand

- Read `references/vit_classification/` when you need the minimal recipe-local integration pattern for config, engine wiring, and tests.
- Read `references/vit_classification/tests/test_vit_gradient_checkpointing.py` when you need a concrete test example.
