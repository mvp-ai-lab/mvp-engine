---
name: gradient-checkpointing
description: Add gradient checkpointing (activation checkpointing) support to any model in this repo. Use when creating a new model, adding gradient checkpointing to an existing model, or when the user mentions gradient checkpointing, activation checkpointing, or memory optimization for training.
---

# Gradient Checkpointing

Add gradient checkpointing (activation checkpointing) support to any model. Core idea: **do not try to write a single generic wrapper**; instead, adapt minimally to each model’s Encoder forward logic.  
**中文：** [SKILL.zh-CN.md](SKILL.zh-CN.md)

## Core Concept

Gradient checkpointing trades compute for memory: activations are not stored in the forward pass and are recomputed during backward. The adaptation is to wrap each layer’s forward call inside the Encoder’s per-layer loop with `torch.utils.checkpoint.checkpoint`.

## Adaptation Workflow

For each new model, do these 3 steps:

### Step 1: Add checkpointing state on the Encoder

In the Encoder’s `__init__` (the module that contains the per-layer loop), add two attributes:

```python
self.gradient_checkpointing = False
self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
```

### Step 2: Wrap each layer call in Encoder.forward

Locate the per-layer `for` loop and, inside it, decide whether checkpointing is enabled. **Key pattern**: define a `custom_forward` closure that takes only tensors that require gradients as arguments; capture everything else in the closure.

```python
use_gc = self.gradient_checkpointing and self.training

for layer in self.layers:
    if use_gc:
        def custom_forward(hidden_states):
            return layer(hidden_states, attention_mask=attention_mask, ...)[0]

        hidden_states = self._gradient_checkpointing_func(custom_forward, hidden_states)
    else:
        hidden_states = layer(hidden_states, attention_mask=attention_mask, ...)[0]
```

**Rules**:
- `custom_forward`’s explicit arguments must be only tensors that need `requires_grad=True` (e.g. `hidden_states`).
- Non-differentiable inputs (`attention_mask`, `rotary_pos_emb`, etc.) are captured in the closure.
- When checkpointing is on, force `output_attentions=False` (attention weights are not saved).
- Condition: `self.gradient_checkpointing and self.training` (no checkpointing at eval).

**If the model has multiple layer types** (e.g. regular layer + mixture layer), implement a checkpointing branch or helper per type. See [references/example-tomatovit.md](references/example-tomatovit.md) (or [references/example-tomatovit.zh-CN.md](references/example-tomatovit.zh-CN.md)) for `_forward_single_branch_layer` and `_forward_mixture_layer`.

### Step 3: Expose enable/disable at the top-level Model

Pick one of two approaches:

**Option A — HuggingFace PreTrainedModel (recommended)**  
If the model subclasses `PreTrainedModel`, set the class attribute:

```python
class MyPreTrainedModel(PreTrainedModel):
    supports_gradient_checkpointing = True
```

`PreTrainedModel.gradient_checkpointing_enable()` will then propagate to submodules and set `gradient_checkpointing = True` and `_gradient_checkpointing_func`.

**Option B — Plain nn.Module**  
Implement enable/disable manually:

```python
class MyModel(nn.Module):
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        gc_kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        self.encoder.gradient_checkpointing = True
        self.encoder._gradient_checkpointing_func = functools.partial(
            torch.utils.checkpoint.checkpoint, **gc_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing = False
```

### Enabling from the Engine

In the recipe engine’s `prepare_model()`, after freezing and before FSDP/DDP wrapping:

```python
gc_enabled = OmegaConf.select(self.config, "model.gradient_checkpointing.enabled", default=False)
gc_use_reentrant = OmegaConf.select(self.config, "model.gradient_checkpointing.use_reentrant", default=False)
if gc_enabled:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc_use_reentrant}
    )
```

YAML config:

```yaml
model:
  gradient_checkpointing:
    enabled: true
    # use_reentrant: false  # default false; usually no need to set
```

## Common Pitfalls

1. **`use_reentrant=False`** (recommended default): works with non-deterministic ops and `torch.compile`, but the checkpointed function’s output must be differentiable w.r.t. its inputs.
2. **Do not pass non-differentiable tensors as explicit arguments to `custom_forward`**: leads to unnecessary recompute or errors.
3. **`output_attentions` conflicts with checkpointing**: recompute does not store intermediate results, so attention weights are lost; when checkpointing is on, force `output_attentions=False`.
4. **Order**: freeze → gradient checkpointing → FSDP/DDP wrap.

## Testing

Write 3 tests per model. See [references/test-patterns.md](references/test-patterns.md) (or [references/test-patterns.zh-CN.md](references/test-patterns.zh-CN.md)) for full templates.

Summary:
1. `test_gradient_checkpointing_enable_sets_state` — enable/disable correctly set encoder state.
2. `test_encoder_uses_checkpointing` — the checkpoint function is actually called during training.
3. `test_gradient_matches_without_checkpointing` — gradients match with and without checkpointing.

## Full Reference

- TomatoViT full adaptation: [references/example-tomatovit.md](references/example-tomatovit.md) | [中文](references/example-tomatovit.zh-CN.md)
- Test templates and full examples: [references/test-patterns.md](references/test-patterns.md) | [中文](references/test-patterns.zh-CN.md)
