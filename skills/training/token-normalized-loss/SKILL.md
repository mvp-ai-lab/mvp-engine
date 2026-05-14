---
name: token-normalized-loss
description: Understand, modify, or port token-normalized training loss for recipes that accumulate unreduced per-token loss across micro-batches and data-parallel ranks, normalize gradients by global supervised token count, and log token-level loss and throughput.
---

# Token-Normalized Loss

## Goal

This skill covers recipe-local token-normalized loss for any token training recipe, not only VLMs. Use it in two modes:

- **Basic VLM mode:** If the target recipe is `basic_vlm` or derived from it, do not reimplement token-normalized loss. Treat this skill as a design map for the existing implementation and modify only the requested layer.
- **Porting mode:** If another recipe still uses batch-mean loss scaling, use this skill to add unreduced per-token loss, global supervised-token normalization, and token-level logging.

The target gradient objective is `sum(per_token_loss) / global_supervised_tokens` over each accumulation window. Keep the implementation recipe-local unless the user explicitly asks for shared engine behavior.

## Existing Reference Implementation

Reference in `basic_vlm`:

- `recipes/basic_vlm/model/qwen3_vl.py` patches Qwen3-VL forward to return unreduced per-token CE loss.
- `recipes/basic_vlm/engine/basic_vlm_engine.py` counts shifted supervised tokens, accumulates loss sums, rescales gradients, and logs token metrics.
- `mvp_engine/utils/metrics.py` provides `DistributedMetricAccumulator` for accumulation-window reductions.
- `recipes/basic_vlm/model/packing/` matters when token-normalized loss interacts with packed attention metadata.

## Workflow

### 1. Identify The Starting Point

Search the target recipe:

```bash
rg -n "reduction=\"none\"|DistributedMetricAccumulator|effective_token|loss_sum|backward_loss_divisor|gradient_scale|tokens/effective" recipes/<recipe> mvp_engine
```

- If the recipe already follows `basic_vlm`, read the relevant existing layer and make a local change.
- If the recipe has no token-normalized loss, use `basic_vlm` as a concrete implementation reference while keeping the target recipe's model and batch schema.
- For non-VLM recipes, keep the same training-accounting pattern but ignore VLM-specific fields such as packed attention or image grids.

### 2. Model Loss Shape

The model must return unreduced per-token loss when labels are present.

Guidelines:

- If the model already returns unreduced loss, keep it and document the expected shape.
- If the model returns scalar mean loss, first check whether it exposes a loss hook such as `model.loss_function`.
- If a hook is not enough, add a recipe-local forward wrapper or injection.
- For causal language models, shift labels by padding one ignored token on the right and taking `[..., 1:]`.
- Use `F.cross_entropy(..., ignore_index=-100, reduction="none")`.
- Chunk logits when full `[batch, sequence, vocab]` projection is too memory-heavy.
- Preserve inference/generation behavior when `labels is None`.

Reference in `basic_vlm`:

- `recipes/basic_vlm/model/qwen3_vl.py` owns `_shift_labels` and `inject_sum_loss_forward`.

### 3. Token Counting

Count tokens before device transfer in `train_pre_step`.

Required counts:

- `total_token_num`: valid tokens from `attention_mask.sum()`;
- `effective_token_num`: shifted-label supervised positions where label is not `-100`.

Reference in `basic_vlm`:

- `recipes/basic_vlm/engine/basic_vlm_engine.py` computes these counts at the start of `train_pre_step`, before moving tensors to device.

### 4. Accumulation Metrics

Register and update accumulation-window metrics:

- `total_token_count`: accumulate `sum`, reduce `sum`;
- `effective_token_count`: accumulate `sum`, reduce `sum`;
- `loss_sum`: accumulate `sum`, reduce `sum`.

Reset the accumulator after each completed optimizer step.

Reference in `basic_vlm`:

- `recipes/basic_vlm/engine/basic_vlm_engine.py` initializes and updates `DistributedMetricAccumulator`.
- `mvp_engine/utils/metrics.py` defines the reusable accumulator.

### 5. Backward And Gradient Rescale

In `backward_step`:

- advance gradient accumulation and set `ctx.should_sync`;
- sum the unreduced micro-batch loss;
- divide by a fixed provisional denominator before backward;
- store the denominator in `ctx.outputs`;
- update local metrics with detached loss sum and token counts.

At the sync optimizer step:

- reduce accumulated metrics;
- require `global_effective_token_count > 0`;
- unscale the optimizer before editing gradients;
- multiply each gradient by:

```python
gradient_scale = (
    float(backward_loss_divisor)
    * float(self.dp_world_size)
    / float(global_effective_token_count)
)
```

Keep gradient clipping after this rescale and before `scaler.step(...)`.

Reference in `basic_vlm`:

- `recipes/basic_vlm/engine/basic_vlm_engine.py` owns `backward_step` and `optimizer_step`.

### 6. Logging

Log only after an optimizer step completes.

Required logs:

- `train/loss = global_loss_sum / global_effective_token_count`;
- `tokens/total`;
- `tokens/effective`;
- `perf/toks_per_sec`.

Preserve existing recipe-specific logs, learning-rate logs, checkpoint behavior, and MFU logging.

Reference in `basic_vlm`:

- `recipes/basic_vlm/engine/basic_vlm_engine.py` owns `train_post_step` token logs.

## Validation

- Confirm the model returns unreduced per-token loss when `labels` is present and keeps inference behavior when labels are absent.
- Confirm `effective_token_num` matches shifted-label supervised positions.
- Confirm one accumulation window logs `train/loss = global_loss_sum / global_effective_token_count`.
- Confirm gradients are rescaled after distributed reduction and before clipping.
- For packed recipes, confirm packed masks and labels do not create cross-source supervised targets.
- Run the smallest available syntax or smoke validation for the changed recipe files.

## Output

- State whether you modified existing `basic_vlm` behavior or ported token-normalized loss into another recipe.
- State whether the model already had unreduced loss support or needed a recipe-local forward patch.
- Summarize how global token counts enter gradient scaling and logging.
- State what validation ran and what remains unverified.
