---
name: token-normalized-loss
description: Add, review, update, and validate token-normalized training loss
  using TokenNormedLossKit, including unreduced per-token loss patching,
  accumulation-window global supervised-token normalization, gradient rescaling,
  and token-level logging.
---

# Token-Normalized Loss

## Goal

Use `TokenNormedLossKit` for the standard objective:

```text
objective = sum(per_token_loss) / global_supervised_tokens
```

Do not reimplement distributed loss/token accumulation in each recipe unless the
kit cannot represent the training loop.

## Required Inputs

- target recipe path;
- model forward path and loss shape;
- batch fields for total and supervised token counts;
- engine `forward_step()`, `backward_step()`, `optimizer_step()`, and
  `train_post_step()` paths;
- gradient accumulation and data-parallel process group;
- recipe-local structure/smoke tests.

## Workflow

### 1. Initialize The Kit

```python
from mvp_engine.kit import TokenNormedLossKit

self.token_loss_kit = TokenNormedLossKit(
    device=self.device,
    data_parallel_world_size=self.data_parallel_world_size,
    token_stats_world_size=self.token_stats_world_size,
    token_stats_group=self.token_stats_group,
)
```

### 2. Ensure Unreduced Per-Token Loss

Prefer:

```python
model = self.token_loss_kit.apply_chunked_token_loss_patch(model)
```

Use recipe-local loss code only when the model is not compatible with the
chunked causal-LM patch.

### 3. Accumulate And Backward

In `backward_step()`:

- advance gradient accumulation and set sync state;
- compute `local_micro_loss_sum = outputs["loss"].sum()`;
- read `effective_tokens` and `total_tokens` from the batch;
- call `accumulate_microbatch(...)`;
- backward the returned provisional loss.

### 4. Reduce And Rescale

At the synchronized optimizer step:

```python
stats = self.token_loss_kit.reduce_window()
self.scaler.unscale_(self.optimizer)
self.token_loss_kit.rescale_gradients(self.model.parameters(), stats)
```

Then clip, step optimizer/scheduler, log reduced stats, and call `reset()`.

## Validation

### Soft Validation

- model returns unreduced per-token loss when labels are present;
- `effective_tokens` matches shifted supervised labels after masking and
  packing;
- all microbatches in one accumulation window use the same backward divisor;
- `reduce_window()` happens once per synchronized optimizer step;
- `rescale_gradients()` runs after unscale and before clipping/optimizer step;
- logs use `TokenLossStats.global_*` values.

### Hard Validation

Run:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

Add impact validation only when proving numerical equivalence across variable
token counts or ranks is part of the task.

## Output

- State whether the model used `apply_chunked_token_loss_patch` or a fallback.
- State where token counts are produced, accumulated, reduced, and logged.
- Report validation and any distributed-runtime gap.

## Read On Demand

- `skills/kit/token-loss-kit/SKILL.md`: kit API guide.
- `references/loss_accounting.md`: objective, gradient scaling, and logging
  rules.
