---
name: fsdp2-prefetching
description: Add, review, update, and validate recipe/model-local FSDP2
  custom prefetching for models whose wrapped module execution order needs
  explicit forward or backward prefetch edges.
---

# FSDP2 Prefetching

## Goal

Add a model-local FSDP2 prefetch hook without changing model math:

- bind a callable on the top-level model class as
  `APPLY_FSDP2_CUSTOM_PREFETCHING`;
- install only prefetch edges between real FSDP2-wrapped modules;
- keep wiring local to the recipe or model implementation;
- avoid YAML toggles, generic graph DSLs, and repo-wide runtime helpers.

The repo runtime contract is fixed: `mvp_engine/distributed/fsdp2.py` calls
`model.__class__.APPLY_FSDP2_CUSTOM_PREFETCHING(model)` after FSDP2 wrapping.

## Required Inputs

Identify these before editing:

- target recipe path;
- top-level model class used by training;
- model builder and `prepare_model()` path;
- FSDP2 wrap targets from `_no_split_modules` or
  `parallel.backend_kwargs.fsdp2.target_classes`;
- top-level `forward()` and key submodule `forward()` execution order;
- whether the model has branches, cross-layer jumps, shared blocks, mixture
  layers, TP class attributes, or high-precision FSDP2 module groups;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if the real wrapped modules or execution order cannot be
derived from the recipe.

## Workflow

### 1. Locate FSDP2 Integration

Search the recipe first:

```bash
rg -n "APPLY_FSDP2_CUSTOM_PREFETCHING|fully_shard|parallelize_model|_no_split_modules" recipes/<recipe>
rg -n "target_classes|set_modules_to_.*prefetch" recipes/<recipe>
```

Find:

- where the model class is defined or subclassed;
- where FSDP2 wrapping is triggered through `parallelize_model(...)`;
- which module classes are wrapped;
- whether TP attributes such as `TP_MODULE_CONFIG` already live on the same
  top-level model class.

### 2. Map Runtime Order

Read the real forward path and write down the wrapped modules in execution
order. Use the actual module instances that exist after construction, not names
guessed from config.

Only include modules wrapped by FSDP2. Do not add edges to helper modules,
unwrapped projections, dataclass containers, or modules skipped by
`ignore_modules`.

Read `references/prefetch_rules.md` when branch transitions or sequential edge
selection is not obvious.

### 3. Add The Hook

Add the smallest recipe/model-local callable near the model class:

```python
def apply_fsdp2_custom_prefetching_for_<model_name>(model: nn.Module) -> None:
    if getattr(model, "_fsdp2_prefetching_configured", False):
        return

    ...
    current_layer.set_modules_to_forward_prefetch([next_layer])
    next_layer.set_modules_to_backward_prefetch([current_layer])

    model._fsdp2_prefetching_configured = True
```

Bind it on the top-level class that training actually instantiates:

```python
class <TopModelClass>(...):
    APPLY_FSDP2_CUSTOM_PREFETCHING = apply_fsdp2_custom_prefetching_for_<model_name>
```

If the top-level class already carries TP or other runtime class attributes,
merge this attribute onto that same class. Do not create a parallel wrapper
class with the same purpose.

### 4. Keep The Wiring Local

Use explicit loops and branches that mirror the model layout. Avoid:

- `torch.fx`, tracing, or graph analysis;
- config-driven prefetch edge lists;
- generic prefetch registries or DSLs;
- edits to `mvp_engine/` unless the user explicitly asks for runtime changes.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- the hook is bound as `APPLY_FSDP2_CUSTOM_PREFETCHING` on the real top-level
  model class;
- FSDP2 still wraps the expected module classes;
- all prefetch targets are real wrapped modules and follow actual forward order;
- backward edges follow the reverse dependency order and are intentionally
  minimal;
- the hook is idempotent and safe to call twice;
- TP and FSDP2 class attributes are merged on the same model class when both are
  present;
- no YAML toggle, generic prefetch DSL, graph helper, or `mvp_engine/` runtime
  change was added;
- structure or smoke checks are not reported as completed prefetch-edge impact
  validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/fsdp2-prefetching/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

Run smoke with an FSDP2-active config or override, for example a shard mesh
greater than one.

Add optional impact validation when the task requires proof of installed edge
identity. Use a recipe-local file such as:

```text
recipes/<recipe>/tests/skills/fsdp2-prefetching/test_prefetch_edges_impact.py
```

The impact test should build the real FSDP2-wrapped model, resolve expected
module paths to live module objects, and compare forward/backward prefetch
targets by object identity.

## Output

- State which model file and top-level class were updated.
- State which callable was bound.
- Summarize the main forward and backward prefetch edges.
- Report soft validation and hard validation status.
- Call out any remaining gap, such as no FSDP2 smoke run or no edge-identity
  impact validation.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/prefetch_rules.md`: edge selection rules and compact examples.
