# Gradient Sync And Optimizer Order

Use when the recipe customizes optimizer steps, loss scaling, gradient clipping,
or mixed precision.

This file owns ordering around `sync_cp_grads(...)` in recipe optimizer steps.
It does not own the runtime definition of CP gradients, attention behavior, or
parity artifact format.

## Invariant

- Unscale first when an AMP scaler is used.
- Token/global loss rescale runs before CP gradient sync.
- `sync_cp_grads(model)` runs before clipping and optimizer step.
- No second independent TP/CP sync path updates the same parameter.

## Public Validation

- Structure tests may confirm `sync_cp_grads` is imported.
- Contract tests must assert relative order inside `optimizer_step`.
- Smoke hooks should verify `_cp_grad_sync` exists when CP grad sync is enabled.

## Assertion Hooks

Use `assert_optimizer_order_contract(...)` for AST-based ordering inside
`optimizer_step`, and `assert_before_train_end(...)` in smoke tests to check that
the runtime attached CP gradient sync when enabled.

## Validation Targets

- Loss matches but gradients differ by context size.
- Clipping happens before CP gradients are summed.
- `_cp_grad_sync` is attached but never called.
