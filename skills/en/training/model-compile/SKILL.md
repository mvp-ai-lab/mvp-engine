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

Add recipe-local tests under `recipes/<recipe>/skill_tests/model-compile/`:

- `test_spec.yaml`: declare the required test layers for this applied skill.
- `test_structure.py`: at least verify recipe import, registry wiring, config
  schema validation, required slots, and logger/checkpoint hooks; it must also
  verify the compile config keys exist and compile is wired at the intended point.
- `test_runtime.py`: at least build dataset, collator, model, optimizer,
  scheduler, and engine successfully without starting training; it must also
  verify `torch.compile` is called only on the intended training-path module(s).
- `test_smoke.py`: cover one real recipe-owned single step: forward, loss,
  backward, optimizer step, logger write, and checkpoint noop or temporary
  save; it must also verify both compile-on and compile-off paths complete
  through the recipe's own training path.
- `test_smoke.py` must use the full real capability path for this skill: real
  engine, real recipe entrypoints, and the real `torch.compile` / logger /
  checkpoint wiring under test. Do not short-circuit it with monkeypatch-based
  fake compile wrappers, fake training steps, or similar test-only stand-ins.
- If the recipe's full-capability single step only makes sense on GPU or
  distributed hardware, write the smoke test as a real launcher-driven smoke
  test and set `gpu_preferred: true` in `test_spec.yaml`; do not degrade it
  into fake logic just to make it run in a weaker environment.

If GPU is available, ask the user whether to run the following tests:
- a single-GPU or single-process `forward/backward` smoke test.
- compare compile on/off loss and logs. Bitwise identity is not required, but there should be no obvious divergence.

Use the user's real recipe/model entrypoints with a minimal recipe-owned config
or batch. Do not substitute an unrelated tiny model for compile validation.

When executing this skill for a user recipe, add these tests automatically. Do not
wait for the user to request test scaffolding separately. Run validation only in
fresh subagents with `fork_context=false`. Do not run these
`python -m tests.test_skills` commands from the main agent's local terminal,
background terminal sessions, or any other non-subagent shell fallback. First run
`python -m tests.test_skills --recipe <recipe> --skill model-compile --layer structure`,
then a new subagent for `--layer runtime` only after structure passes, and then a
new subagent for `--layer smoke` only after runtime passes. The main agent should
summarize all three layer results. If `test_smoke.py` is blocked by GPU,
distributed-launch requirements, or execution permissions, the main agent should
return the exact `python -m tests.test_skills` command and any required launch
command instead.

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
