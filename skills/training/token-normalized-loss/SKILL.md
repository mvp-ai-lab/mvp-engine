---
name: token-normalized-loss
description: Add, review, update, and validate recipe-local token-normalized training loss for token models using unreduced per-token loss, accumulation-window global supervised-token normalization, gradient rescaling, and token-level logging.
---

# Token-Normalized Loss

## Goal

Add token-normalized training loss without changing the intended objective:

```text
objective = sum(per_token_loss) / global_supervised_tokens
```

The denominator is computed across the full gradient-accumulation window and
all data-parallel ranks. Keep the implementation recipe-local unless the user
explicitly asks for shared engine behavior.

## Required Inputs

Identify these before editing:

- target recipe path;
- model forward path and current loss shape;
- batch schema for `input_ids`, `labels`, `attention_mask`, packing metadata, or
  equivalent token fields;
- engine `train_pre_step()`, `forward_step()`, `backward_step()`,
  `optimizer_step()`, and `train_post_step()` paths;
- gradient accumulation and data-parallel topology;
- existing metric accumulator or distributed reduction helper;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if the supervised-token definition or packed-label semantics
cannot be derived from the recipe.

## Workflow

### 1. Locate Runtime Integration Points

Search the recipe first:

```bash
rg -n "reduction=\"none\"|effective_token|loss_sum|backward_loss_divisor|gradient_scale|tokens/effective|DistributedMetricAccumulator" recipes/<recipe>
```

Find:

- where model loss is computed;
- where labels are shifted or masked;
- where token counts are available before device transfer;
- where micro-batch loss is passed to backward;
- where optimizer-step logging happens.

### 2. Return Unreduced Per-Token Loss

The model must return unreduced supervised-token loss when labels are present.

Common causal-LM shape:

```python
loss = F.cross_entropy(
    logits[..., :-1, :].contiguous().view(-1, vocab_size),
    labels[..., 1:].contiguous().view(-1),
    ignore_index=-100,
    reduction="none",
)
```

Preserve inference behavior when labels are absent. If full logits are too
large, use the recipe's existing chunking or projection strategy.

### 3. Count Tokens

Count token metrics once per micro-batch before backward:

- `total_token_num`: valid input tokens, often `attention_mask.sum()`;
- `effective_token_num`: supervised shifted-label positions where label is not
  `-100`.

For packed batches, count supervised labels after packing/truncation and ensure
labels do not create cross-sample targets.

### 4. Accumulate Loss And Token Metrics

Across one gradient-accumulation window, accumulate locally and reduce globally:

- total token count: local sum, distributed sum;
- effective token count: local sum, distributed sum;
- loss sum: local sum, distributed sum.

Use an existing recipe accumulator when available. Keep the accumulation boundary
aligned with the optimizer step.

### 5. Backward And Rescale Gradients

In `backward_step()`:

- advance gradient accumulation and set `ctx.should_sync`;
- sum the unreduced micro-batch loss;
- divide by a fixed provisional denominator before backward;
- store that denominator for the optimizer step;
- update detached loss/token metrics.

At the synchronized optimizer step:

- reduce accumulated loss/token metrics;
- require `global_effective_token_count > 0`;
- unscale the optimizer before editing gradients;
- multiply gradients by the final token-normalization factor;
- clip gradients after this rescale and before `scaler.step(...)`.

Read `references/loss_accounting.md` for the exact accounting formula and order.

### 6. Log Token Metrics

Log only after an optimizer step completes:

- `train/loss = global_loss_sum / global_effective_token_count`;
- `tokens/total`;
- `tokens/effective`;
- `perf/toks_per_sec`.

Preserve existing recipe logs, learning-rate logs, checkpoint behavior, and MFU
logging.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- model returns unreduced per-token loss when labels are present;
- inference/generation behavior is unchanged when labels are absent;
- `effective_token_num` matches shifted supervised labels after masking,
  truncation, and packing;
- loss sum, total token count, and effective token count accumulate over the same
  optimizer-step window;
- distributed reductions use sum for loss and token counts;
- gradient rescale happens after metric reduction and unscale, but before
  clipping and optimizer step;
- zero supervised-token windows fail or follow the recipe's existing zero-token
  policy explicitly;
- `train/loss`, token counters, and throughput use global reduced values;
- no repo-wide token-normalized loss logic was added to `mvp_engine/`;
- structure or smoke checks are not reported as completed normalization-impact
  validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/token-normalized-loss/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

Add optional impact validation only when the task requires proof of numerical
equivalence. Use a recipe-local file such as:

```text
recipes/<recipe>/tests/skills/token-normalized-loss/test_normalization_impact.py
```

The impact test should compare a controlled accumulation window against the
direct objective `sum(per_token_loss) / global_supervised_tokens`, including
variable token counts across micro-batches or ranks when feasible.

## Output

- State whether the model already returned unreduced loss or needed a patch.
- State where supervised tokens are counted and where metrics are reduced.
- State where gradient rescaling happens relative to unscale, clipping, and
  optimizer step.
- Report soft validation and hard validation status.
- Call out any remaining gap, such as no distributed environment for multi-rank
  token normalization validation.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/loss_accounting.md`: objective, gradient scaling, and logging rules.
