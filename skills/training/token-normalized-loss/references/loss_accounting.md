# Token-Normalized Loss Accounting

Use this reference when implementing or reviewing gradient and logging
accounting.

## Objective

For one optimizer step:

```text
loss = sum(per_token_loss over all micro-batches and DP ranks)
       / global_supervised_tokens
```

The supervised-token denominator should count only positions that contribute to
loss after shifting, masking, truncation, packing, and ignore-index handling.

## Unreduced Loss

Use unreduced per-token loss:

```python
per_token_loss = F.cross_entropy(
    logits,
    labels,
    ignore_index=-100,
    reduction="none",
)
```

The model or forward wrapper may flatten or chunk logits, but it must preserve
one loss value per supervised position before summing in the engine.

## Provisional Backward Denominator

Backward can run before the final global token count is known by using a fixed
provisional denominator:

```python
loss = local_micro_loss_sum / backward_loss_divisor
loss.backward()
```

At the synchronized optimizer step, after reducing global token counts:

```python
gradient_scale = (
    backward_loss_divisor
    * data_parallel_world_size
    / global_effective_token_count
)
```

Multiply every non-None gradient by `gradient_scale`.

## Operation Order

At sync step:

1. reduce accumulated `loss_sum`, `total_token_count`, and
   `effective_token_count`;
2. require positive `global_effective_token_count`;
3. unscale optimizer gradients;
4. multiply gradients by `gradient_scale`;
5. clip gradients;
6. optimizer step, scaler update, scheduler step;
7. log global token metrics;
8. reset accumulation metrics.

## Logging

Use reduced global values:

```python
logs["train/loss"] = global_loss_sum / global_effective_token_count
logs["tokens/total"] = global_total_token_count
logs["tokens/effective"] = global_effective_token_count
logs["perf/toks_per_sec"] = global_total_token_count / step_time_seconds
```

Do not log per-rank or per-micro-batch token-normalized loss as the optimizer-step
training loss unless the metric name explicitly says so.
