---
name: loss-spike-guard
description: Add a recipe-local loss spike guard that skips anomalous micro-batches. Use when a recipe should detect scalar-loss or per-token-loss spikes during training, expose guard thresholds in config, and zero the current micro-batch contribution without moving guard logic into mvp_engine core.
---

# Loss Spike Guard

## Goal

- Add a recipe-local guard that detects anomalously high training loss and skips the current micro-batch contribution.
- Support both scalar mean-loss training and unreduced per-token loss training.
- Keep the guard disabled by default and controlled by recipe config.

## Required Inputs

- The target recipe path under `recipes/<recipe>/`.
- The recipe engine file, especially `prepare_optimizer()` and `backward_step()`.
- The recipe config schema and YAML configs.
- The exact YAML config(s) where the guard should be enabled, if any.
- The current loss shape:
  - scalar loss: `outputs["loss"]` is a scalar tensor used directly for backward
  - per-token loss: `outputs["loss"]` is unreduced and the engine computes `loss_sum` plus supervised token counts
- The device used for distributed scalar reductions.

## Workflow

### 1. Gather Context

- Read the engine and config schema before editing.
- Search for existing guard or spike support:

```bash
rg -n "LossGuard|PerTokenLossGuard|loss_spike|spike_multiplier|loss_sum|effective_token" recipes/<recipe> mvp_engine
```

- Do not change `mvp_engine/`; this guard is recipe-local unless the user explicitly asks otherwise.

### 2. Add Guard Config

- Add optimizer config keys:

```python
loss_spike_skip_multiplier: float | None = Field(None, gt=0.0)
loss_spike_skip_window_size: int = Field(8, ge=1)
loss_spike_skip_min_history: int = Field(3, ge=1)
```

- Keep `loss_spike_skip_multiplier: null` or omitted as the disabled default.
- Only enable the guard in YAML configs where the recipe intentionally wants spike skipping.
- Do not stop at the schema: inspect the recipe's YAML configs and add the keys to the target training config when the guard should be active for that run.
- For staged recipes, enable the guard only in the intended stage config and leave unrelated stages unchanged.
- Do not add explicit disabled keys to unrelated YAML configs just to expose the schema defaults.
- When reproducing an existing recipe behavior, preserve the existing enabled values exactly instead of replacing them with defaults or `null`.

### 3. Add The Scalar Guard

- Add a recipe-local helper such as `recipes/<recipe>/guards/loss.py`.
- Implement `LossGuard` with:
  - a fixed-size history window
  - a minimum warmup history before skipping
  - a `spike_multiplier`
  - no skip when `spike_multiplier is None`
- On a spike, log a compact warning that includes current loss, baseline loss, history size, loss factor, and multiplier.
- Update history for warmup and normal non-spike losses.
- Do not add spike losses to history.

### 4. Add The Per-Token Guard When Needed

- If the recipe uses unreduced per-token loss, add `PerTokenLossGuard(LossGuard)`.
- The per-token guard must accept:
  - local `loss_sum`
  - local supervised `token_count`
  - `step`
  - `device`
- Convert `loss_sum` and `token_count` to a float64 tensor pair on the target device.
- If distributed is initialized, `all_reduce` the pair with `SUM`.
- If the global token count is zero, do not skip.
- Pass `global_loss_sum / global_token_count` into the scalar guard and include `global_token_count` in the warning.
- Do not make the scalar guard consume unreduced loss tensors directly.

### 5. Wire The Engine

- In `prepare_optimizer()` or another initialization point after config validation, create the guard:
  - scalar loss path: `LossGuard(...)`
  - per-token loss path: `PerTokenLossGuard(...)`
- Store it on the engine as `self.loss_guard`.
- In `backward_step()`, call the guard before backward and before final metric updates for the current micro-batch.

For scalar loss:

```python
loss = outputs["loss"] / self.config.optim.gradient_accumulation_steps
if self.loss_guard.check(outputs["loss"], step=int(self.step)):
    loss = loss * 0.0
    outputs["logs"]["train/loss"] = 0.0
ctx.loss = loss
```

For per-token loss:

```python
loss = local_micro_loss_sum / float(backward_loss_divisor)
if self.loss_guard.check(
    local_micro_loss_sum,
    micro_effective_token_count,
    step=int(self.step),
    device=self.device,
):
    loss = loss * 0.0
    local_micro_loss_sum = local_micro_loss_sum * 0.0
```

- Preserve existing gradient accumulation, scaler, clipping, optimizer, scheduler, and logging behavior.
- In the per-token path, do not set the micro-batch effective token count to zero unless the existing recipe already treats skipped samples as removed from the denominator.
- Do not add new all-zero-token optimizer-step behavior for the guard; keep the recipe's existing zero-token handling.

## Validation

- Confirm the guard is disabled when `loss_spike_skip_multiplier` is `None`.
- Confirm warmup losses fill history without skipping.
- Confirm a loss greater than `baseline * spike_multiplier` is skipped once enough history exists.
- Confirm non-spike losses update history.
- For per-token loss, confirm the guard computes `global_loss_sum / global_token_count` and returns no skip when token count is zero.
- Run the smallest available syntax or smoke validation for the changed recipe files.
- Add recipe-local tests only when higher-priority user or repository instructions allow tests.

## Output

- State whether the scalar or per-token guard path was used.
- State which guard, engine, config, and YAML files were updated.
- Summarize how a skipped micro-batch is zeroed.
- State what validation ran and what remains unverified.

## Read On Demand

- For a current VLM per-token implementation example, inspect `recipes/basic_vlm/guards/loss.py`,
  `recipes/basic_vlm/configs/schema.py`, `recipes/basic_vlm/configs/stage3.yaml`, and the
  `PerTokenLossGuard` wiring in `recipes/basic_vlm/engine/basic_vlm_engine.py`.
