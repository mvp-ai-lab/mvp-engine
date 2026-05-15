---
name: gradient-checkpointing
description: Add, review, update, and validate gradient checkpointing for mvp-engine recipes. Use when wiring model.gradient_checkpointing config, enabling model-native checkpointing, or adding minimal manual activation checkpointing to repeated blocks.
---

# Gradient Checkpointing

## Goal

Add recipe-local gradient checkpointing without changing model math:

- keep checkpointing disabled by default and enabled explicitly through config;
- prefer model-native support over manual wrapping;
- enable checkpointing before FSDP, DDP, TP, or other distributed wrapping;
- validate that the real training path can run with checkpointing enabled.

## Required Inputs

Identify these before editing:

- target recipe path;
- config schema and YAML configs;
- model builder and engine `prepare_model()` path;
- top-level model class and repeated-block owner;
- whether the model already supports `gradient_checkpointing_enable(...)`;
- distributed wrapping entrypoint such as `parallelize_model(...)`;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if the model entrypoint, intended config, or validation
environment cannot be derived from the recipe.

## Workflow

### 1. Locate Runtime Integration Points

Search the recipe first:

```bash
rg -n "gradient_checkpoint|checkpoint\\(|parallelize_model|prepare_model|use_cache|output_attentions" recipes/<recipe>
```

Find:

- where the model is built;
- where distributed wrapping happens;
- where config is parsed;
- which module owns the repeated layer/block loop;
- whether the model already has checkpointing flags or methods.

### 2. Choose The Implementation Path

Use the native-support path when the model already routes repeated blocks
through a checkpoint function. Typical signs:

- `gradient_checkpointing_enable(...)` and `gradient_checkpointing_disable(...)`;
- `supports_gradient_checkpointing = True`;
- `self.gradient_checkpointing` gates layer execution;
- `_gradient_checkpointing_func` or equivalent is already used in the layer loop.

Use the manual path only when the real repeated-block loop has no checkpointing
support. Keep manual changes inside the recipe model files.

Read `references/patterns.md` before manual adaptation or when the native path is
not obvious.

### 3. Add Config

Expose this shape under `model`:

```yaml
model:
  gradient_checkpointing:
    enabled: false
    use_reentrant: false
```

Add the matching typed schema or config class. Prefer `use_reentrant: false`
unless the target model requires reentrant checkpointing.

### 4. Wire Native Model Support

For models that already support checkpointing, enable it in `prepare_model()`
after model construction and before distributed wrapping:

```python
gc_config = self.config.model.gradient_checkpointing
if gc_config.enabled:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc_config.use_reentrant}
    )

model = parallelize_model(...)
```

Do not rewrite model internals on this path.

### 5. Add Manual Model Support

When native support is absent:

- add `gradient_checkpointing_enable(...)` and
  `gradient_checkpointing_disable(...)` to the top-level recipe model or the
  block-loop owner;
- store `self.gradient_checkpointing` and `self._gradient_checkpointing_func`;
- route repeated blocks through the checkpoint function only when
  `self.training` and checkpointing are both true;
- pass only gradient-carrying tensors as explicit checkpoint arguments;
- capture masks, position ids, RoPE tensors, cache objects, and flags in a
  closure;
- gate or disable incompatible outputs such as KV cache and attentions.

Keep parameter names, checkpoint keys, and forward math unchanged.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- the chosen native/manual path matches the model's real capabilities;
- config exposes `model.gradient_checkpointing.enabled` and `use_reentrant`;
- config defaults keep checkpointing off and an explicit override can enable it;
- engine wiring enables checkpointing before distributed wrapping;
- optimizer construction still sees the same trainable parameters;
- `use_cache` or auxiliary outputs do not conflict with checkpointing;
- manual checkpointing, if added, is limited to repeated blocks and does not pass
  non-differentiable objects as explicit checkpoint inputs;
- checkpointing does not alter parameter names, checkpoint loading, forward
  outputs, or loss math;
- no repo-wide wrapper was added to `mvp_engine/`;
- CPU-only or structure-only checks are not reported as completed runtime
  checkpoint validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/gradient-checkpointing/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q --config-override model.gradient_checkpointing.enabled=true
```

If the recipe smoke test does not expose `--config-override`, set the equivalent
recipe-local smoke override before running it.

## Output

- State whether native support or manual adaptation was used.
- State which config, engine, and model files changed.
- State where checkpointing is enabled and where distributed wrapping happens.
- Report soft validation and hard validation status.
- Call out any remaining gap, such as no GPU/NPU environment for runtime smoke.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/patterns.md`: native-support and manual-adaptation code patterns.
