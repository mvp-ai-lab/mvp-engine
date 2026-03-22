---
name: gradient-checkpointing
description: Add gradient checkpointing (activation checkpointing) to a recipe in this repo. Use when checking whether a model already supports checkpointing, wiring recipe config and engine toggles, or adding minimal model-side checkpoint logic and tests.
---

# Gradient Checkpointing

Add gradient checkpointing without introducing a repo-wide wrapper.  
**中文：** [SKILL.md](../../../zh-cn/training/gradient-checkpointing/SKILL.md)

## Goal

- Enable gradient checkpointing on the target recipe.
- Keep model math unchanged.
- Add recipe-local config, engine wiring, and verification tests.

## 1. Classify the model before editing

- Prefer the existing-support path.
- Use the manual-adaptation path only when the model does not already checkpoint its repeated blocks.

### Existing-support path

Use this path when all of the following are true:

- The top-level model exposes `gradient_checkpointing_enable()` and `gradient_checkpointing_disable()`.
- The model propagates `gradient_checkpointing` and `_gradient_checkpointing_func` to the modules that need them.
- The repeated blocks already route through `_gradient_checkpointing_func` when `self.gradient_checkpointing and self.training` is true. In recent `transformers` models this is often implemented by `GradientCheckpointingLayer.__call__`.

### Manual-adaptation path

Use this path when the repeated compute blocks do not already checkpoint themselves.

## 2. Existing-support path: only wire the recipe

- Do not rewrite model internals if the model already supports checkpointing.
- In `prepare_model()`, enable checkpointing after building the model and before FSDP/DDP/TP wrapping:

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

- Under the new config system, add `model.gradient_checkpointing` to the recipe's `ConfigClass` / schema, then read it via typed attribute access in the engine.

- Prefer `use_reentrant: false` unless the target model specifically requires reentrant checkpointing.

### ViT reference

- `recipes/vit_classification` is the canonical simple path.
- The HuggingFace ViT layers already inherit `GradientCheckpointingLayer`, so the example only changes the recipe engine, config, and tests.
- Archived reference files live under `references/vit_classification/`.

## 3. Manual-adaptation path: patch the module with the layer loop

- On the module that owns the per-layer loop, add:

```python
self.gradient_checkpointing = False
self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
```

- In the loop, call each layer through `_gradient_checkpointing_func` when training with checkpointing enabled.
- Pass only gradient-carrying tensors as explicit checkpoint arguments. Capture masks, rope embeddings, and other non-differentiable inputs in a closure.
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

## 4. Archive the example instead of committing recipe-only churn

- If you temporarily modify a demo recipe to validate the workflow, move the final changed files into `references/<recipe>/`.
- Restore `recipes/<recipe>/` to the clean branch state before committing the skill.
- Archive only the files that actually changed for checkpointing.

## 5. Testing

Add recipe-local tests that cover:

1. enable and disable toggles set the expected module state.
2. the checkpoint function is actually invoked during training.
3. gradients match with and without checkpointing.

Reference tests:

- `references/vit_classification/tests/test_vit_gradient_checkpointing.py`

## Pitfalls

- Do not add a repo-wide generic wrapper.
- Do not pass non-differentiable inputs as explicit checkpoint arguments.
- Enable checkpointing before distributed wrapping.
- For `transformers` models, do not manually rewrap layers that already use `GradientCheckpointingLayer`.

## Reference

- ViT minimal recipe integration: `references/vit_classification/`
