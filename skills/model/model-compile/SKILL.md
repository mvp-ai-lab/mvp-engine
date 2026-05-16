---
name: model-compile
description: Add, review, update, and validate torch model.compile support for mvp-engine recipes. Use when wiring model compile config, choosing compile scope, placing compilation before distributed wrapping, or creating recipe-local validation for compile wiring.
---

# Model Compile

## Goal

Add recipe-local `model.compile` support without changing model math:

- keep compilation disabled by default unless the recipe intentionally enables it;
- compile only modules on the real training hot path;
- place compilation before FSDP, DDP, TP, or other distributed wrapping unless a
  documented recipe constraint requires otherwise;
- validate that the real training path can run with compile enabled.

## Required Inputs

Identify these before editing:

- target recipe path;
- config schema and YAML configs;
- model builder and engine `prepare_model()` path;
- candidate module or callable on the real training hot path;
- distributed wrapping entrypoint such as `parallelize_model(...)`;
- known graph-break regions, dynamic-shape paths, data-dependent Python, or
  unsupported custom ops;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if the intended compile target, validation environment, or
performance goal cannot be derived from the recipe.

## Workflow

### 1. Locate Runtime Integration Points

Search the recipe first:

```bash
rg -n "torch\\.compile|\\.compile\\(|compile_backend|compile_mode|compiler\\.disable|parallelize_model|prepare_model" recipes/<recipe>
```

Find:

- where the model is built;
- where distributed wrapping happens;
- which modules execute on every training step;
- whether the top-level forward includes Python-heavy recipe glue;
- whether any branches, teacher models, EMA models, or auxiliary heads execute.

### 2. Choose Compile Scope

Default to one compile-friendly hot-path target:

- compile the top-level model only when its forward is mostly tensor/model math;
- compile a core submodule when top-level forward builds tokens, masks, packed
  metadata, multimodal glue, or other Python-heavy inputs;
- evaluate teacher, EMA, auxiliary, or distillation branches separately;
- do not compile many tiny modules without evidence that it helps.

Read `references/patterns.md` when the scope or graph-break handling is not
obvious.

### 3. Add Config

Expose compile config under `model`:

```yaml
model:
  compile: false
  compile_backend: "inductor"
  compile_mode: "default"
```

Add the matching typed schema or config class. Keep the default conservative
unless the recipe already intentionally enables compile.

Config meaning:

- `model.compile`: boolean feature switch. Use `false` as the safest default for
  new recipes because compile can add first-step latency, expose graph breaks,
  and require accelerator-specific validation. Set `true` only when the recipe
  has a known compile-safe target or the user explicitly wants compile enabled.
- `model.compile_backend`: backend passed to `torch.compile(..., backend=...)`.
  Prefer `"inductor"` for normal CUDA/GPU training because it is PyTorch's
  default production compiler backend. Use `"aot_eager"` or `"eager"` only for
  debugging compiler capture and graph-break behavior. Custom backend strings or
  callables are recipe-specific and should be documented where configured.
- `model.compile_mode`: mode passed to `torch.compile(..., mode=...)`.
  Supported PyTorch modes include `"default"`, `"reduce-overhead"`,
  `"max-autotune"`, and `"max-autotune-no-cudagraphs"`. Use `"default"` first;
  use `"reduce-overhead"` when Python/CUDA graph overhead dominates and static
  shapes make cudagraph-style capture practical; use `"max-autotune"` only when
  longer compile time is acceptable for potential steady-state speedups; use
  `"max-autotune-no-cudagraphs"` when autotuning is desired but cudagraphs are
  unsafe for the recipe.

### 4. Wire Compile

Compile after model construction and recipe-local patches, before distributed
wrapping:

```python
if self.config.model.compile:
    model.compile(
        backend=self.config.model.compile_backend,
        mode=self.config.model.compile_mode,
    )

model = parallelize_model(...)
```

If only a submodule is safe to compile, compile that submodule explicitly and
document the scope in code or in the change summary.

Keep parameter names, checkpoint keys, forward outputs, and loss math unchanged.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- config exposes `model.compile`, `compile_backend`, and `compile_mode`;
- compile defaults are intentional and an explicit override can enable/disable
  compilation;
- compile scope matches the real training hot path;
- top-level compile is avoided or graph breaks are isolated when recipe glue is
  Python-heavy or data-dependent;
- compile happens before distributed wrapping, or the exception is documented;
- teacher, EMA, auxiliary, or distillation branches were evaluated explicitly;
- checkpoint format, parameter names, forward outputs, and loss math are
  unchanged;
- CPU-only or structure-only checks are not reported as completed runtime
  compile validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/model-compile/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q --config-override model.compile=true
```

If the recipe smoke test does not expose `--config-override`, set the equivalent
recipe-local smoke override before running it.

Add impact validation:

```text
recipes/<recipe>/tests/skills/model-compile/test_compile_performance.py
```

The impact test should compare controlled compile-off and compile-on runs,
separate first-step compile latency from steady-state timing, and use
recipe-appropriate thresholds. For example, after the first step compile overhead, the compile-on run should not be slower than compile-off, and may be faster by a significant margin depending on the recipe and target.

## Output

- State which config, engine, and model files changed.
- State which module or callable is compiled.
- State where compilation happens relative to distributed wrapping.
- Report soft validation and hard validation status.
- Call out any remaining gap, such as no GPU/NPU environment for runtime smoke
  or no steady-state performance measurement.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/patterns.md`: compile-scope, placement, and graph-break patterns.
