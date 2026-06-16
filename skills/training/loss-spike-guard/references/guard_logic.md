# Loss Spike Guard Logic

Use this reference when implementing or reviewing skip semantics.

## Scalar Loss

The scalar guard keeps a fixed-size history of non-spike loss values:

```python
baseline = sum(loss_history) / len(loss_history)
is_spike = current_loss > baseline * spike_multiplier
```

Rules:

- return `False` when `spike_multiplier is None`;
- append warmup losses until `min_history` is reached;
- skip only after enough history exists;
- append normal non-spike losses;
- do not append spike losses;
- log current loss, baseline, history size, loss factor, and multiplier when
  skipping.

## Per-Token Loss

Use a per-token guard only when the recipe computes unreduced loss and owns a
supervised token count.

Inputs:

- local loss sum;
- local supervised token count;
- current step;
- device for reduction tensors.

Rules:

- stack `[loss_sum, token_count]` as float64 on the target device;
- when distributed is initialized, all-reduce the pair with `SUM`;
- return `False` when global token count is zero;
- compare `global_loss_sum / global_token_count` through the scalar guard;
- include global token count in skip diagnostics.

Do not feed unreduced tensors directly into the scalar guard.

## Engine Wiring

Call the guard before backward:

```python
if not self.loss_kit.guard_loss(loss_value, step=int(self.step)):
    loss = loss * 0.0
```

For per-token loss handled by `TokenNormedLossKit`, call
`self.token_loss_kit.guard_loss(loss_sum, token_count, step=int(self.step))`.

For per-token loss, zero the local loss sum used by metric accumulation when the
micro-batch is skipped. Keep token counts unchanged unless the recipe's existing
denominator semantics explicitly remove skipped samples.

Do not alter gradient accumulation, optimizer stepping, scaler updates,
scheduler stepping, or zero-token handling only to support the guard.
