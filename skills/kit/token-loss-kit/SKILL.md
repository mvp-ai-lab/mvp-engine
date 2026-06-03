---
name: token-loss-kit
description: Use TokenNormedLossKit for unreduced per-token loss patching,
  accumulation-window global token normalization, gradient rescaling, and token
  loss logging in MVP-Engine recipes.
---

# Token Loss Kit

## Goal

Use `TokenNormedLossKit` for the standard token-normalized objective:

```text
sum(per_token_loss) / global_supervised_tokens
```

The kit provides:

- `apply_chunked_token_loss_patch(...)`;
- `accumulate_microbatch(...)`;
- `reduce_window()`;
- `rescale_gradients(...)`;
- `reset()`.

## Required Inputs

- recipe engine forward/backward/optimizer-step hooks;
- batch fields for total and supervised token counts;
- data-parallel world size and process group;
- model compatibility with the chunked causal-LM loss patch;
- gradient accumulation divisor.

## Workflow

1. Create the kit in engine init with the local reduction device and DP group.
2. Patch the model if it does not already return unreduced per-token loss.
3. In `backward_step()`, sum the microbatch loss and call
   `accumulate_microbatch(...)`.
4. At synchronized optimizer step, call `reduce_window()`, unscale the optimizer,
   then call `rescale_gradients(...)`.
5. Clip gradients, step optimizer/scheduler, log reduced loss and token counts,
   then call `reset()`.

## Validation

### Soft Validation

- model loss is unreduced per supervised token;
- `effective_tokens` matches shifted labels after masking and packing;
- all microbatches in one accumulation window use the same backward divisor;
- gradient rescale happens after reduction and unscale, before clipping and
  optimizer step;
- logging uses reduced global stats.

### Hard Validation

Run recipe structure and smoke tests. Add impact tests only when numerical
equivalence must be proven across variable token counts or ranks.

## Output

- State where the kit is initialized, where loss is accumulated, and where
  gradients are rescaled.
- State token-count fields and validation status.

## Read On Demand

- `skills/training/token-normalized-loss/SKILL.md`: feature-oriented workflow
  for adding this behavior to a recipe.
