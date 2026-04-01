---
name: model-compile
description: Add or adjust model.compile for a recipe, decide compile scope and placement, wire config under model, and validate correctness and performance.
---

# Model Compile

## Goal

- Add or adjust `model.compile` support for a training recipe under `recipes/<recipe>/`.
- Keep compile disabled by default and enabled explicitly through config.
- Compile the modules that matter on the real training hot path, and keep compile before `parallelize_model` unless there is a documented exception.

## Required Inputs

- The target recipe path and the recipe's `prepare_model()` implementation.
- The candidate modules on the real training hot path.
- Whether the recipe also has teacher models, EMA modules, auxiliary heads, or other independent branches.
- The target recipe's config or schema files.
- GPU availability if correctness or performance validation should be run.

## Workflow

### 1. Gather context first

- Find the recipe's `prepare_model()` and confirm the base model construction is complete.
- Read the reference implementation under `references/` when it matches the target recipe.
- Search the repo for nearby compile precedents:

```bash
rg -n "torch\.compile|model\.compile|compile_backend|compile_mode" recipes
```

### 2. Decide the compile scope

- Compile only modules on the training hot path.
- If the top-level `forward()` mixes Python-heavy preprocessing, token building, positional setup, or other recipe glue, do not compile the whole model by default.
- When the recipe has teacher, EMA, auxiliary heads, or distillation branches, evaluate them separately instead of hiding them inside the main model decision.
- Prefer one compile-friendly core target over fragmenting compile across many tiny child modules.

### 3. Decide compile placement

- Default order is:
  - call `model.compile(...)`
  - then call `parallelize_model(...)`
- If a recipe needs another order, document the reason in code comments or in the change summary.

### 4. Implement config and code

- Put compile config under `model`:
  - `model.compile`
  - `model.compile_backend`
  - `model.compile_mode`
- Expose those keys through the recipe schema or `ConfigClass`.
- Wire compile in `prepare_model()` with a pattern like:

```python
if self.config.model.compile:
    model.compile(
        backend=self.config.model.compile_backend,
        mode=self.config.model.compile_mode,
    )
```

- Keep `model.compile` defaulted to `False`.
- Do not change checkpoint format, parameter names, or the public interface just to fit compile.

### 5. Validate correctness and performance

- At minimum, validate config parsing and compile wiring.
- If GPU is available, ask the user whether to run:
  - a single-process or single-GPU forward/backward smoke test
  - a compile-on vs compile-off comparison for loss and logs
- Record first-step compile latency, whether steady state is reached, throughput changes, and memory changes when those measurements are available.

## Validation

- `model.compile`, `model.compile_backend`, and `model.compile_mode` are wired into config.
- The compiled target matches the real training hot path.
- Compile is not fragmented into many tiny child modules without evidence.
- Compile placement is either the default order or a documented exception.
- Extra branches such as teacher or EMA paths were evaluated explicitly.

## Output

- State which model, engine, and config files were updated.
- State which module or callable is being compiled.
- State the chosen compile order and any reason for deviating from the default.
- Summarize what correctness or performance validation ran and what remains unverified.

## Read On Demand

- Read `references/vit_classification/configs/train.yaml` and `references/vit_classification/engine/vit_classification_engine.py` when you need the current reference implementation for compile wiring.
