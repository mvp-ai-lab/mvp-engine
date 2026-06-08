---
name: context-parallel
description: Add, review, update, and validate recipe-local long-context attention
  using mvp-engine's context mesh and yunchang USP/Ring attention, with zigzag
  ring layout as the default target for new recipes. Use when a recipe needs
  sequence/context parallel attention beyond tensor-mesh sequence parallelism.
---

# Context Parallel

## Goal

Add long-context attention without changing model math:

- enable `parallel.mesh.context`;
- configure `parallel.backend_kwargs.long_context`;
- keep attention-module rewrites local to the recipe/model;
- shard sequence inputs, labels, positions, and masks across context ranks with
  `mvp_engine.kit.CPKit.prepare_causal_batch`;
- keep context ranks on the same samples and micro-batches;
- validate context-local gradient synchronization;
- coordinate CP grad hooks with TP hooks when both touch the same parameter.

The repo runtime contract is fixed:
`mvp_engine/distributed/parallelize.py` initializes yunchang process groups,
then calls `model.__class__.APPLY_LONG_CONTEXT_ATTENTION(model, device_mesh,
long_context_config)` before TP/FSDP2 wrapping. When `grad_sync=true`, replicated
parameter grads are summed across the context mesh after FSDP2 wrapping.

## Required Inputs

Identify these before editing:

- target recipe path;
- top-level model class used by training;
- attention module classes and Q/K/V/out projection shapes;
- hidden-state layout and sequence dimension;
- RoPE/position-id/mask handling;
- whether attention uses MHA, GQA, MQA, packed QKV, varlen, or packed samples;
- dataset/sampler rank logic and data-parallel mesh dimensions;
- intended `context`, `ulysses_degree`, `ring_degree`, and `ring_impl_type`;
- active TP/FSDP2 mesh settings;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask only if context size, target model, or available devices cannot be derived.

## Workflow

### 1. Locate Attention And Data Sharding

Search the recipe first:

```bash
rg -n "APPLY_LONG_CONTEXT_ATTENTION|LongContext|parallelize_model|parallel.mesh|long_context" recipes/<recipe>
rg -n "q_proj|k_proj|v_proj|out_proj|scaled_dot_product_attention|flash_attn|attention" recipes/<recipe>
rg -n "DistributedSampler|get_world_size|get_rank|get_data_parallel|device_mesh|RuntimeContext" recipes/<recipe>
```

Find the real top-level class, attention modules, and sampler/loader code.

### 2. Configure Mesh And Backend

Use a separate context mesh dimension:

```yaml
parallel:
  mesh:
    shard: <fsdp2 shard size>
    context: <ulysses_degree * ring_degree>
    tensor: <tp size>
  backend_kwargs:
    long_context:
      enabled: true
      ulysses_degree: <U>
      ring_degree: <R>
      ring_impl_type: zigzag
      attn_impl: fa
```

Rules:

- prefer `ring_impl_type=zigzag` for new long-context training recipes;
- use `basic` only for bootstrap/debug paths or recipes that explicitly require
  contiguous rank-local positions;
- `context > 1` is required when enabled;
- `shard > 1` is required because this repo rejects pure model parallelism
  without FSDP2;
- `sequence_parallel` and `long_context.enabled` must not both be true;
- all ranks that differ only by `context` must read the same samples;
- exclude `tensor` and `context` from data-loader sharding and global batch
  accounting.

### 3. Add A Recipe-Local Attention Hook

Bind one hook on the real top-level model class:

```python
def apply_long_context_attention_for_<model>(model, device_mesh, long_context_config) -> None:
    ...
    model._long_context_attention_configured = True


class <TopModelClass>(...):
    APPLY_LONG_CONTEXT_ATTENTION = apply_long_context_attention_for_<model>
```

Inside the hook:

- replace only the recipe attention modules that can consume sequence-local
  tensors;
- instantiate yunchang attention through
  `mvp_engine.distributed.cp.build_long_context_attention`;
- keep Q/K/V as `[batch, local_seq, heads, head_dim]`;
- return outputs as `[batch, local_seq, hidden]`;
- handle GQA by preserving smaller KV heads when the selected kernel supports it;
- make the hook idempotent.

### 4. Review Boundaries

Read `references/cp_rules.md` before finalizing. Check:

- input ids, masks, position ids, RoPE caches, and packed metadata are sharded
  with the same configured layout through the CP kit;
- next-token labels are shifted globally before layout extraction; never do a
  local-only label shift for `zigzag`;
- causal attention receives correct global positions for local tokens;
- loss normalization accounts for local tokens across the context group, and
  logging reductions use the intended context/data-parallel groups;
- multimodal placeholder spans are either wholly local after extraction or are
  rejected/resampled;
- routers, packing, metrics, and generation paths do not treat local sequence as
  global sequence;
- context grad sync is not duplicated in recipe code;
- when TP also syncs replicated local-activation params, hook ownership is
  explicit and compatible.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- `parallel.mesh.context` and `backend_kwargs.long_context` are present;
- long-context and tensor-mesh `sequence_parallel` are not both enabled;
- dataloading excludes `context` from sample sharding;
- the real top-level class binds `APPLY_LONG_CONTEXT_ATTENTION`;
- engine batch preparation uses `mvp_engine.kit.CPKit.prepare_causal_batch`;
- attention receives and returns local-sequence tensors;
- RoPE/position/mask/loss boundaries are explicitly handled for `zigzag`;
- recipe code does not rely on `get_basic_sequence_offset` unless it also has an
  explicit non-basic path;
- no yunchang import happens on normal non-long-context runs.
- CP and TP grad hooks do not independently delta-sync the same parameter.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/context-parallel/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q --run-smoke \
  --world-size 4 \
  --config-override parallel.mesh.shard=2 \
  --config-override parallel.mesh.context=2 \
  --config-override parallel.backend_kwargs.long_context.enabled=true \
  --config-override parallel.backend_kwargs.long_context.ulysses_degree=1 \
  --config-override parallel.backend_kwargs.long_context.ring_degree=2 \
  --config-override parallel.backend_kwargs.long_context.ring_impl_type=zigzag \
  --config-override optim.mixed_precision=bf16
```

Also run a `basic` smoke only when the recipe keeps a basic compatibility path.
Add optional impact validation when correctness depends on attention parity,
multimodal spans, packing boundaries, or gradient accumulation.

## Output

- State which model/config/test files changed.
- Summarize context, Ulysses, and Ring degrees.
- State which attention modules were replaced.
- Report structure, smoke, and optional impact validation status.
- Call out missing runtime dependencies such as `yunchang` or `flash-attn`.

## Read On Demand

- `references/cp_rules.md`: context mesh, yunchang, boundary, and
  validation rules.
- `references/asserts.py`: recipe-local assertion template.
