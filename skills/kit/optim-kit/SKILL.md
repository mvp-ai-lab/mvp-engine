---
name: optim-kit
description: Use OptimKit to build optimizers over trainable parameters and
  Transformers learning-rate schedulers in MVP-Engine recipes.
---

# Optim Kit

## Goal

Use `OptimKit` for standard optimizer and scheduler construction:

- `build_optimizer(model, optimizer, lr, weight_decay, **kwargs)` collects only
  parameters with `requires_grad=True`;
- `build_lr_scheduler(optimizer, lr_scheduler, **kwargs)` delegates scheduler
  construction to Transformers `get_scheduler`.

## Required Inputs

- recipe optimizer config;
- model after freeze policy has been applied;
- total training steps and warmup policy;
- any optimizer-specific kwargs.

## Workflow

1. Apply freeze policy before calling `build_optimizer`.
2. Pass the configured torch optimizer name, learning rate, and weight decay.
3. Build the scheduler after total steps are known.
4. Keep recipe-specific scheduler math, such as warmup-step calculation, in the
   recipe engine.

## Validation

### Soft Validation

- optimizer is built after freeze policy;
- at least one parameter remains trainable;
- scheduler step count matches the recipe loop config;
- optimizer-specific kwargs are intentional.

### Hard Validation

Run recipe structure and smoke tests.

## Output

- State optimizer, scheduler, warmup/step inputs, and validation status.

## Read On Demand

- `mvp_engine/kit/optim/__init__.py` for the exact API.
