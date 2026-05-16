---
name: tensor-parallel
description: Add, review, update, and validate recipe-local tensor-parallel
  plans, TP metadata postprocessors, and compatible mesh config for mvp-engine
  models.
---

# Tensor Parallel

## Goal

Add tensor parallelism without changing model math:

- define a recipe/model-local `TP_MODULE_CONFIG`;
- bind it on the top-level model class as `TP_MODULE_CONFIG`;
- add `TP_MODULE_POSTPROCESSORS` only when sharding changes module-local runtime
  metadata;
- set a compatible `parallel.mesh.tensor` layout;
- keep model-specific TP logic in the recipe/model implementation.

The repo runtime contract is fixed: `mvp_engine/distributed/tp.py` reads
`model.__class__.TP_MODULE_CONFIG`, applies `parallelize_module(...)`, then runs
optional `TP_MODULE_POSTPROCESSORS`.

## Required Inputs

Identify these before editing:

- target recipe path;
- top-level model class used by training;
- model builder and `prepare_model()` path;
- repeated module classes containing TP-covered linears;
- direct child names of target `nn.Linear` modules;
- forward code that reshapes, splits, indexes, or caches head/expert metadata;
- current `parallel.mesh` values and intended TP size;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if TP size, available devices, or target module ownership
cannot be derived from the task.

## Workflow

### 1. Locate Runtime Integration

Search the recipe first:

```bash
rg -n "TP_MODULE_CONFIG|TP_MODULE_POSTPROCESSORS|parallelize_model|parallel.mesh" recipes/<recipe>
rg -n "q_proj|k_proj|v_proj|out_proj|fc1|fc2|gate_proj|up_proj|down_proj" recipes/<recipe>
```

Find:

- where the top-level model class is defined or subclassed;
- where `parallelize_model(...)` is called;
- which repeated block classes contain attention, MLP, projector, expert, or
  head linears;
- whether FSDP2 prefetching or other runtime class attributes already live on
  the same top-level class.

### 2. Draft The TP Plan

Build `TP_MODULE_CONFIG` as:

```python
MODEL_TP_MODULE_CONFIG: dict[str, object] = {
    "<RuntimeBlockClass>": {
        "q_proj": "col",
        "k_proj": "col",
        "v_proj": "col",
        "o_proj": "row",
    },
}
```

Use runtime class names as keys. Use direct child linear names as plan keys.
Plan values should be `"col"` or `"row"` unless the recipe has a documented
custom `ParallelStyle`.

Read `references/tp_rules.md` when mapping attention, MLP, VLM projector, MoE,
or metadata-sensitive modules.

### 3. Bind The Plan

Bind the plan on the top-level class that training actually instantiates:

```python
class <TopModelClass>(...):
    TP_MODULE_CONFIG = MODEL_TP_MODULE_CONFIG
```

If the top-level class already carries `APPLY_FSDP2_CUSTOM_PREFETCHING` or other
runtime class attributes, merge `TP_MODULE_CONFIG` onto that same class. Do not
create a second wrapper class with the same purpose.

### 4. Add Postprocessors Only When Needed

Add `TP_MODULE_POSTPROCESSORS` only when module-local metadata must change after
TP sharding. Common examples:

- attention modules cache `num_heads`, `num_key_value_heads`, or `all_head_size`;
- MoE or routed modules cache local expert counts;
- forward code uses `view`, `reshape`, `split`, loops, or indexing based on
  global head/expert/hidden dimensions.

Keep postprocessors local and idempotent. Mutate module runtime fields, not the
global config.

### 5. Update Mesh Config

Set `parallel.mesh.tensor` to the desired TP size. In this repo, pure TP without
FSDP2 is rejected by `parallelize_model(...)`, so `parallel.mesh.shard` must be
greater than one when `tensor > 1`.

Preserve the intended data-parallel product:

```yaml
parallel:
  mesh:
    replicate: <D>
    shard: <S>
    tensor: <T>
```

## Validation

### Soft Validation

Review the modified recipe without running tests:

- `TP_MODULE_CONFIG` is bound on the real top-level model class;
- config keys match runtime class names and plan keys match direct child
  modules;
- plan values are valid TP modes and match expansion/merge semantics;
- all metadata-sensitive modules were reviewed for postprocessing;
- postprocessors, if present, are idempotent and mutate only local runtime
  metadata;
- TP, FSDP2 prefetching, and other class attributes are merged on the same
  top-level class when used together;
- mesh `replicate`, `shard`, and `tensor` are compatible with the intended world
  size;
- no generic TP DSL, YAML plan, or `mvp_engine/` runtime change was added;
- structure or smoke checks are not reported as completed sharding-impact
  validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/tensor-parallel/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q --config-override parallel.mesh.tensor=2
```

Also supply a compatible shard override if the base config has `shard: 1`.

Add optional sharding impact validation when the task requires proof of local
shard shapes. Use a recipe-local file such as:

```text
recipes/<recipe>/tests/skills/tensor-parallel/test_sharding_impact.py
```

The impact test should compare TP-covered parameter local shapes against
pre-parallel reference shapes under the active tensor mesh.

Add optional numerical impact validation when the task requires proof that TP
preserves model semantics. Use a recipe-local file such as:

```text
recipes/<recipe>/tests/skills/tensor-parallel/test_loss_parity_impact.py
```

The impact test should run the same deterministic batch through TP-off and
TP-on models and compare loss or logits within recipe-appropriate tolerances.
Use eval mode, fixed seeds, no optimizer step, identical weights, and the same
mixed precision policy where feasible.

## Output

- State which model and config files changed.
- Summarize the TP plan by runtime module class.
- State whether TP postprocessors were added and why.
- State final mesh settings.
- Report soft validation and hard validation status.
- Call out any remaining gap, such as no TP-active smoke run, no local-shape
  impact validation, or no loss-parity validation.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/tp_rules.md`: TP plan, postprocessor, mesh, and impact rules.
