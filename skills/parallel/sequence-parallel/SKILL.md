---
name: sequence-parallel
description: Add, review, update, and validate recipe-local sequence-parallel
  plans that reuse the tensor-parallel mesh in mvp-engine models.
---

# Sequence Parallel

## Goal

Add Megatron-style sequence parallelism for models that already use tensor
parallelism:

- enable `parallel.backend_kwargs.sequence_parallel`;
- keep `parallel.mesh.tensor` as the SP size;
- do not add a separate sequence mesh dimension;
- keep model-specific sequence-parallel plans in recipe/model code;
- bind optional sequence-parallel metadata on the top-level model class.

The repo runtime contract is fixed:
`mvp_engine/distributed/tp.py` reads `model.__class__.TP_MODULE_CONFIG`, and when
sequence parallel is enabled it also merges optional
`model.__class__.SEQUENCE_PARALLEL_MODULE_CONFIG`.

## Required Inputs

Identify these before editing:

- target recipe path;
- top-level model class used by training;
- existing `TP_MODULE_CONFIG` and `TP_MODULE_POSTPROCESSORS`;
- norm, dropout, residual, and activation modules that should run on sequence
  shards;
- tensor kwargs passed into those modules, such as masks, scales, position ids,
  or cache tensors;
- sequence dimension used by the model's hidden states;
- dataset or dataloader sharding code, including any `RuntimeContext`,
  `DataLoadMesh`, sampler, or rank/world-size logic;
- current `parallel.mesh` values and intended TP/SP size;
- sequence length or packing/collation rule, and whether it is divisible by
  TP/SP size;
- derived sequence lengths from routing, top-k, packing, or cache slicing;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if TP/SP size, sequence dimension, available devices, or
target module ownership cannot be derived from the task.

## Workflow

### 1. Locate TP Wiring

Search the recipe first:

```bash
rg -n "TP_MODULE_CONFIG|SEQUENCE_PARALLEL|parallelize_model|parallel.mesh|sequence_parallel" recipes/<recipe>
rg -n "RuntimeContext|DataLoadMesh|device_mesh|dp_dims|DistributedSampler|rank|world_size" recipes/<recipe>
rg -n "LayerNorm|RMSNorm|Dropout|dropout|norm" recipes/<recipe>
```

Find:

- where the top-level model class is defined or subclassed;
- which repeated block classes already have TP-covered linears;
- which modules consume and return hidden states shaped like `[batch, seq, hidden]`
  or `[seq, batch, hidden]`;
- whether dataloading shards samples by all global ranks or only by data-parallel
  mesh dimensions;
- whether runtime class attributes already live on the same top-level class.

### 2. Confirm Mesh And Backend

Sequence parallel reuses tensor parallelism:

```yaml
parallel:
  mesh:
    shard: <fsdp2 shard size>
    tensor: <tp_sp_size>
  backend_kwargs:
    sequence_parallel: true
```

Rules:

- `parallel.mesh.tensor > 1` is required;
- `parallel.mesh.shard > 1` is required because this repo rejects pure TP/SP
  without FSDP2;
- `sp_size == tp_size`;
- all ranks in the same TP/SP group must read the same samples and
  micro-batches;
- `tensor` is model-parallel, not data-parallel, and must not multiply global
  batch size or dataset slots;
- shard data only over data-parallel dimensions, normally `replicate` and
  FSDP2 `shard`;
- exclude `tensor` and any other non-data-parallel dimensions such as `context`
  from dataloader sharding;
- when using `mvp_dataset`, pass a `device_mesh` plus `dp_dims` that excludes
  `tensor`, or provide an equivalent sampler/loader guarantee;
- prefer `seq_len % parallel.mesh.tensor == 0`; otherwise pad/mask explicitly
  and verify every SP gather/scatter/reduce path handles uneven sequence shards;
- check routed, packed, or cached sequence lengths too, not only the raw input
  length;
- do not add `parallel.mesh.sequence`.

### 3. Add Sequence-Parallel Plans

Keep normal TP linears in `TP_MODULE_CONFIG`. Add sequence-sharded modules to
`SEQUENCE_PARALLEL_MODULE_CONFIG`:

```python
MODEL_SEQUENCE_PARALLEL_MODULE_CONFIG: dict[str, object] = {
    "<RuntimeBlockClass>": {
        "input_layernorm": "sequence",
        "post_attention_layernorm": "sequence",
        "dropout": "sequence",
    },
}
```

Bind on the same top-level model class:

```python
class <TopModelClass>(...):
    TP_MODULE_CONFIG = MODEL_TP_MODULE_CONFIG
    SEQUENCE_PARALLEL_MODULE_CONFIG = MODEL_SEQUENCE_PARALLEL_MODULE_CONFIG
    SEQUENCE_PARALLEL_SEQUENCE_DIM = 1
```

Use `SEQUENCE_PARALLEL_SEQUENCE_DIM = 1` for `[batch, seq, hidden]`. Use `0` for
`[seq, batch, hidden]`.

### 4. Review Boundaries

Read `references/sp_rules.md` before finalizing the plan. Check:

- entry modules that receive full hidden states before the first column-parallel
  projection;
- column-parallel projections gather sequence-sharded inputs before computing,
  while row-parallel outputs remain sequence-sharded;
- replicated parameters that consume local sequence shards and therefore need
  tensor-mesh gradient sync;
- FSDP2-wrapped replicated DTensor parameters without a `"tensor"` mesh dim,
  which need post-accumulate sync of the newly added `.grad` delta;
- whether grad sync hooks are installed at the right lifecycle point without
  applying tensor-mesh all-reduce twice;
- row-parallel outputs that remain sequence-sharded;
- norms and dropout that are safe on local sequence shards;
- tensor kwargs passed into SP-planned modules, because runtime hooks only
  prepare the first positional tensor input;
- any loss, router, sampler, cache, logging, or metric code that needs the full
  sequence.

### 5. Keep Postprocessors Local

Use existing `TP_MODULE_POSTPROCESSORS` only for metadata changed by TP sharding,
such as head counts or hidden dimensions. Sequence parallel usually should not
mutate model config or cached global sequence length.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- `sequence_parallel` is configured under `parallel.backend_kwargs`;
- `parallel.mesh.tensor > 1` and `parallel.mesh.shard > 1` for SP-active smoke;
- no `parallel.mesh.sequence` field was introduced;
- dataloading uses identical samples for ranks that differ only on the `tensor`
  mesh dimension;
- global batch accounting excludes `tensor` and includes only data-parallel
  dimensions such as `replicate` and `shard`;
- `TP_MODULE_CONFIG` remains bound on the real top-level model class;
- `SEQUENCE_PARALLEL_MODULE_CONFIG`, if present, is bound on the same class;
- `SEQUENCE_PARALLEL_SEQUENCE_DIM` matches the model hidden-state layout;
- sequence plan values use `"sequence"` and TP plan values use `"col"` or
  `"row"`;
- entry and output boundaries have been reviewed for full-sequence assumptions;
- tensor kwargs into SP-planned modules are absent, already layout-compatible, or
  handled in recipe code;
- replicated grad sync is installed after FSDP2 when FSDP2 is required, sums
  newly added TP-local grad deltas, and avoids double all-reduce.
- any custom replicated-grad hook is validated with at least two backward calls
  before `zero_grad`, so post-accumulate hooks do not re-reduce old gradients.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/sequence-parallel/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q \
  --config-override parallel.mesh.tensor=2 \
  --config-override parallel.mesh.shard=2 \
  --config-override parallel.backend_kwargs.sequence_parallel=true
```

Also pass `--world-size 4` or another compatible value when the recipe smoke test
requires the world size to match `replicate * shard * tensor`.

Add optional sequence-layout impact validation when the task requires proof that
norm/dropout modules receive sequence-local tensors. Use a recipe-local file
such as:

```text
recipes/<recipe>/tests/skills/sequence-parallel/test_sequence_layout_impact.py
```

The impact test should run with `parallel.backend_kwargs.sequence_parallel=true`
and inspect a stable hook point before/after SP-planned modules. Assert local
hidden-state sequence length is divided across the tensor mesh, while batch and
hidden dimensions remain compatible with the recipe's model layout. If the
recipe uses `[seq, batch, hidden]`, assert that `SEQUENCE_PARALLEL_SEQUENCE_DIM`
is `0`; otherwise assert against the default `[batch, seq, hidden]` layout.

Add optional numerical impact validation when the task requires proof that SP
preserves model semantics. Use a recipe-local file such as:

```text
recipes/<recipe>/tests/skills/sequence-parallel/test_loss_parity_impact.py
```

The impact test should run the same deterministic batch through TP/SP-off and
TP/SP-on models and compare loss or logits within recipe-appropriate tolerances.
Use eval mode, fixed seeds, no optimizer step, identical weights, identical
packing/collation, and the same mixed precision policy where feasible. Gather or
reduce global token counts before loss/metric normalization when the loss path
consumes sequence-local tensors.

Add optional replicated-gradient impact validation when replicated parameters
consume sequence-sharded activations, especially norms or dropout-adjacent
modules. Use a recipe-local file such as:

```text
recipes/<recipe>/tests/skills/sequence-parallel/test_replicated_grad_impact.py
```

The impact test should compare TP/SP-off and TP/SP-on gradients for replicated
parameters, and should cover at least two backward calls before `zero_grad` when
post-accumulate hooks are used.

Add a dataloader identity impact test when changing recipe data loading. Within
one TP/SP group, assert sample ids or a stable batch fingerprint are identical
across tensor ranks, while data-parallel ranks receive distinct slots. Use a
recipe-local file such as:

```text
recipes/<recipe>/tests/skills/sequence-parallel/test_dataloader_identity_impact.py
```

## Output

- State which model, config, and test files changed.
- Summarize TP/SP size and final mesh settings.
- State how dataloader sharding preserves identical batches within TP/SP
  groups.
- Summarize `SEQUENCE_PARALLEL_MODULE_CONFIG` by runtime module class.
- State whether `SEQUENCE_PARALLEL_SEQUENCE_DIM` was added and why.
- Report soft validation and hard validation status.
- Call out any remaining gap, such as no SP-active smoke run or no parity check.

## Read On Demand

- `references/sp_rules.md`: sequence-parallel layout and boundary rules.
- `references/asserts.py`: recipe-local hard-validation assertion template.
