---
name: model-compile
description: Add or adjust model.compile for a recipe, decide compile scope and placement, wire config under optim, and validate correctness and performance. The files under references/ are reference implementations for this skill. Use for enabling compile on a new model, changing compile order in an existing recipe, or investigating compile regressions.
---

# Model Compile

## Goal

Add or adjust `model.compile` support for a training recipe under `recipes/<recipe>/`, while keeping:
- compile disabled by default and enabled explicitly through config.
- compile applied to the modules actually used during training.
- compile always placed before `parallelize_model`.

## Repo conventions

- Put config keys under `optim`:
  - `optim.compile`
  - `optim.compile_backend`
  - `optim.compile_mode`
- Put compile logic in `prepare_model()` in most cases.
- Do not compile the optimizer, scheduler, or dataloader.

## Workflow

### 1. Gather context first

- Find the recipe's `prepare_model()`. Confirm the base model construction is already complete.
- Read the reference implementations under `references/` first when they match the target recipe. These files are concrete examples of the expected config and engine wiring.
- Search the repo for similar recipes as additional precedents:

```bash
rg -n "torch\\.compile|optim\\.compile|compile_backend|compile_mode" recipes
```

For this skill, `references/vit_classification/configs/train.yaml` and
`references/vit_classification/engine/vit_classification_engine.py` are the current reference implementation.

### 2. Decide compile scope

- Compile only modules on the training hot path.
- Check whether there are teacher, EMA, auxiliary heads, distillation branches, or other independent `forward()` paths. If so, ask whether they all need compile.
- If the top-level `forward()` mixes Python-heavy preprocessing, token building, positional encoding setup, output branching, or other recipe glue, do not compile the whole model by default.
- In that case, ask the user whether to extract one compile-friendly core module/callable that covers the dense tensor hot path.
- Avoid splitting compile across many small child modules unless you have evidence it helps; fragmented compile often loses cross-layer fusion and can greatly increase first-step latency.

### 3. Decide compile placement

Default preference:
- call `model.compile(...)` first, then `parallelize_model(...)`.

Hard requirement:
- If you do not use the default order, explain the reason in code comments or the change summary.

### 4. Implement config and code

Recommended pattern:

```python
if bool(OmegaConf.select(self.config, "optim.compile", default=False)):
    model.compile(
        backend=OmegaConf.select(self.config, "optim.compile_backend", default="inductor"),
        mode=OmegaConf.select(self.config, "optim.compile_mode", default="default"),
    )
```

Rules:
- `optim.compile` must have a default of `False`.
- Read `backend` and `mode` through `OmegaConf.select(..., default=...)`.
- Compile extra modules such as teacher or EMA separately; do not hide them inside the main-model logic.
- If you need a recipe-specific encoder/core submodule just for compile, prefer one larger target over compiling dozens of blocks individually.
- Do not change checkpoint format, parameter names, or the model's public interface just to fit compile.

### 5. Validate

At minimum:
- config validation

If GPU is available, ask the user whether to run the following tests:
- a single-GPU or single-process `forward/backward` smoke test.
- compare compile on/off loss and logs. Bitwise identity is not required, but there should be no obvious divergence.

Good to record:
- first-step compile latency.
- whether step 2 / steady-state is reached at all; a compile that only finishes step 1 is not usable.
- steady-state throughput change.
- memory change.

## Acceptance checklist

- [ ] `optim.compile`, `optim.compile_backend`, and `optim.compile_mode` are wired into config.
- [ ] The compiled target module matches the real training hot path.
- [ ] The compile target is not over-fragmented; one compile-friendly core is preferred over many tiny compiled children.
- [ ] Compile placement has a clear rationale; exception order is documented.
- [ ] Extra modules and branches have each been evaluated for compile.
