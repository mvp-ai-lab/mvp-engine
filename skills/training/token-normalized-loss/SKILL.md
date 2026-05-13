---
name: token-normalized-loss
description: Add token-normalized training loss to a recipe. Use when a recipe should accumulate unreduced per-token loss across micro-batches and data-parallel ranks, normalize gradients by the global supervised token count, and log token-level loss/throughput without moving the pattern into mvp_engine core.
---

# Token-Normalized Loss

## Goal

- Replace batch-mean loss scaling with supervised-token normalization for a recipe-local training loop.
- Keep the implementation recipe-local and preserve unrelated existing recipe behavior.
- Make gradients equivalent to optimizing `sum(per_token_loss) / global_supervised_tokens` over each accumulation window.

## Required Inputs

- The target recipe path under `recipes/<recipe>/`.
- The recipe engine file that implements `train_pre_step`, `forward_step`, `backward_step`, `optimizer_step`, and `train_post_step`.
- The model builder or model wrapper that controls whether forward returns a mean loss or unreduced per-token loss.
- The batch schema, especially `input_ids`, `attention_mask`, `labels`, optional packed-attention fields, and any existing metric fields.
- The data-parallel world size source, usually `get_data_parallel_world_size(self.device_mesh)`.

## Workflow

### 1. Gather Context

- Read the recipe engine, model builder, config schema, and collator before editing.
- Search for existing token-normalized or unreduced-loss support:

```bash
rg -n "reduction=\"none\"|DistributedMetricAccumulator|effective_token|loss_sum|backward_loss_divisor|gradient_scale" recipes/<recipe> mvp_engine
```

- Reuse `mvp_engine.utils.metrics.DistributedMetricAccumulator` if available.
- Do not change `mvp_engine/` unless the user explicitly asks; this pattern belongs in the recipe.

### 2. Make Model Forward Return Per-Token Loss

- If the model already returns unreduced per-token loss, keep it and document the expected shape.
- If the model returns a scalar mean loss, first check whether the model exposes a loss hook such as
  `model.loss_function` or requires a recipe-local forward override.
- Ask the user whether memory-saving loss computation is needed:
  - If no, prefer the minimal hook approach, such as assigning `model.loss_function` to an unreduced CE callable when
    the upstream forward already calls that hook.
  - If yes, inject or wrap the recipe-local forward so training computes `lm_head` logits only for loss chunks and
    avoids materializing full `[batch, sequence, vocab]` logits.
- For causal language models:
  - shift labels by padding one ignored token on the right and taking `[..., 1:]`
  - compute logits only for the hidden states needed by the loss
  - use `F.cross_entropy(..., ignore_index=-100, reduction="none")`
  - chunk logits if the full vocabulary projection is memory-heavy
- Keep generation/inference behavior valid when `labels is None`; forward injection must return normal logits for
  inference/generation paths.

### 3. Count Tokens Before Device Transfer

- In `train_pre_step`, compute:
  - `total_token_num` from `attention_mask.sum()`
  - `effective_token_num` from shifted labels where `label != -100`
- Count shifted labels, not raw labels, so the denominator matches causal LM loss positions.
- Move the new counts through `ctx.data` as Python integers or scalar-safe values.
- Preserve existing batch preparation logic.

Example:

```python
data["total_token_num"] = int(data["attention_mask"].sum().item())
shifted_labels = F.pad(data["labels"], (0, 1), value=-100)[..., 1:]
data["effective_token_num"] = int((shifted_labels != -100).sum().item())
```

### 4. Accumulate Local Metrics

- In the engine constructor, register distributed metrics for at least:
  - `total_token_count`: accumulate `sum`, reduce `sum`
  - `effective_token_count`: accumulate `sum`, reduce `sum`
  - `loss_sum`: accumulate `sum`, reduce `sum`
- Reset the metric accumulator after each completed optimizer step.

### 5. Backward With A Provisional Divisor

- In `backward_step`, advance gradient accumulation before backward and set `ctx.should_sync`.
- Sum the per-token loss for the micro-batch.
- Divide by a fixed provisional denominator before backward so dataloading can overlap with GPU work. Use a denominator that is stable within the accumulation window, such as:

```python
backward_loss_divisor = (
    int(self.config.data.batch_size)
    * int(self.config.data.max_seq_len)
    * int(self.config.optim.gradient_accumulation_steps)
)
loss = local_micro_loss_sum / float(backward_loss_divisor)
```

- Store `backward_loss_divisor` in `ctx.outputs` for the sync step.
- Update accumulated metrics with total tokens, effective tokens, and detached loss sum.

### 6. Rescale Gradients At Sync

- In `optimizer_step`, return early until `ctx.should_sync` is true.
- On sync:
  - reduce the metric accumulator
  - require `global_effective_token_count > 0`
  - unscale the optimizer before gradient edits
  - multiply every gradient by:

```python
gradient_scale = (
    float(backward_loss_divisor)
    * float(self.dp_world_size)
    / float(global_effective_token_count)
)
```

- Keep gradient clipping after this rescale and before `scaler.step(...)`.
- Preserve existing optimizer, scaler, scheduler, zero-grad, and `ctx.optimizer_step_completed` behavior.
- Attach global token counts, global loss sum, and learning rates to `ctx.outputs` for post-step logging.

### 7. Log Token-Normalized Metrics

- In `train_post_step`, log only after an optimizer step completes.

```python
train_loss = float(global_loss_sum / global_effective_token_count)
```

- Add logs for:
  - `train/loss`
  - `tokens/total`
  - `tokens/effective`
  - `perf/toks_per_sec`
  - learning rates, if the recipe already logs them here
- Preserve existing recipe-specific metric logging when present.

## Validation

- Confirm the model returns unreduced per-token loss when `labels` is present and keeps inference behavior when `labels` is absent.
- Confirm `effective_token_num` matches shifted-label supervised positions.
- Confirm one accumulation window logs `train/loss = global_loss_sum / global_effective_token_count`.
- Confirm gradients are rescaled after distributed reduction and before clipping.
- Run the smallest available syntax or smoke validation for the changed recipe files.

## Output

- State which model, engine, and optional config/test files were updated.
- State whether the model already had unreduced loss support or needed a recipe-local forward patch.
- Summarize how global token counts enter gradient scaling and logging.
- State what validation ran and what remains unverified.

## Read On Demand

- For a concrete OpenBee/Qwen3-VL implementation example, read
  `references/openbee-per-token-loss.patch`.
