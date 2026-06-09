---
name: loss-spike-guard
description: Add, review, update, and validate loss spike guards for scalar-loss or per-token-loss training, including config thresholds, micro-batch skip wiring, distributed loss reduction.
---

# Loss Spike Guard

## Goal

Use the optimizer kit loss guard to skip anomalous micro-batch loss
contributions:

- keep the guard disabled by default unless a recipe stage intentionally enables
  it;
- support scalar mean-loss training and unreduced per-token loss training;
- detect spikes only after a warmup history is available;
- zero the skipped micro-batch contribution without changing unrelated gradient
  accumulation, optimizer, scheduler, scaler, or logging behavior.

## Required Inputs

Identify these before editing:

- target recipe path;
- config schema and YAML configs;
- engine `prepare_optimizer()` or initialization path;
- engine `forward_step()` and `backward_step()` loss shape;
- whether `outputs["loss"]` is scalar or unreduced per-token loss;
- if using token-normalized training, where `TokenNormedLossKit` accumulates the
  micro-batch loss sum;
- distributed context used for scalar reductions;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if the intended stage config or loss shape cannot be derived
from the recipe.

## Workflow

### 1. Locate Runtime Integration Points

Search the recipe first:

```bash
rg -n "LossGuard|loss_spike|loss_sum|effective_token|backward_step|gradient_accumulation" recipes/<recipe>
```

Find:

- where optimizer-related state is initialized;
- where micro-batch loss is converted into `ctx.loss`;
- whether token loss is accumulated through `TokenNormedLossKit`;
- where metrics consume loss sums or scalar loss values;
- where distributed token/loss reductions already happen;
- which YAML stage should enable the guard, if any.

### 2. Add Config

Expose guard config under `optim`:

```yaml
optim:
  loss_spike_skip_multiplier: null
  loss_spike_skip_window_size: 8
  loss_spike_skip_min_history: 3
```

Config meaning:

- `loss_spike_skip_multiplier`: disabled when `null`; otherwise skip when
  `current_loss > history_baseline * multiplier`.
- `loss_spike_skip_window_size`: number of non-spike losses retained for the
  moving baseline.
- `loss_spike_skip_min_history`: minimum retained losses before skipping is
  allowed.

Only enable the guard in YAML configs where spike skipping is intended. Do not
add explicit disabled keys to unrelated stage YAMLs just to restate schema
defaults.

### 3. Use Guard Logic

Use the shared loss-kit guard implementations:

- `LossGuard` for scalar losses;
- `PerTokenLossGuard` only when the recipe uses unreduced per-token loss sums.

Read `references/guard_logic.md` before implementing or changing the check
semantics in `mvp_engine/kit/loss/loss.py` or
`mvp_engine/kit/loss/token_loss.py`.

### 4. Wire The Engine

Create the guard during engine initialization, commonly in `prepare_optimizer()`
or another path that runs before the first training step:

```python
self.loss_kit.build_loss_guard(...)
```

For token-normalized per-token loss, use `self.token_loss_kit.build_loss_guard(...)`.

In `backward_step()`, call `self.loss_kit.guard_loss(...)` or
`self.token_loss_kit.guard_loss(...)` before backward and before final metric
updates for the current micro-batch. `guard_loss(...)`
returns `True` when the loss should participate in backward.

For scalar loss, skip by zeroing the backward loss and the logged micro-batch
loss contribution.

For per-token loss, call the guard with local loss sum and local supervised token
count, then zero the local loss sum and backward loss when the guard returns
`False`. Do not set token count to zero unless the recipe already treats skipped
samples as removed from denominators.

When the recipe uses `TokenNormedLossKit`, zero `local_micro_loss_sum` before
calling `TokenNormedLossKit.accumulate_microbatch(...)`; keep token counts
unchanged unless the recipe explicitly removes skipped samples from the
normalization denominator.

### 5. Preserve Training Semantics

Do not change unrelated behavior:

- gradient accumulation step advancement;
- scaler, clipping, optimizer, and scheduler order;
- zero-token behavior;
- metric accumulation boundaries;
- distributed reductions unrelated to the guard.

Spike losses should not update the guard history. Warmup and non-spike losses
should update it.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- config exposes `optim.loss_spike_skip_multiplier`,
  `loss_spike_skip_window_size`, and `loss_spike_skip_min_history`;
- the guard is disabled when multiplier is `None` / `null`;
- warmup losses fill history without skipping;
- spike losses are skipped only after `min_history` is reached;
- spike losses do not update history;
- scalar and per-token loss shapes are handled by the correct guard path;
- per-token guard reduces global `loss_sum` and token count before comparing;
- zero-token micro-batches do not trigger skip;
- skipped micro-batches zero loss contribution before backward and before final
  metric updates;
- no repo-wide loss-guard logic was added to `mvp_engine/`;
- structure or smoke checks are not reported as completed skip-impact validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/loss-spike-guard/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

## Output

- State whether scalar or per-token guard wiring was used.
- State which config, YAML, guard, and engine files changed.
- State where skipped micro-batches are zeroed.
- Report soft validation and hard validation status.
- Call out any remaining gap, such as no distributed environment for per-token
  global reduction validation.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/guard_logic.md`: scalar and per-token guard behavior rules.
