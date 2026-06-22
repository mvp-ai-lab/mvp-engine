---
name: context-parallel
description: Add, review, update, and validate recipe-local context-parallel
  training using mvp-engine's context mesh, Ulysses attention, CPKit batch
  helpers, and explicit CP gradient synchronization.
---

# Context Parallel

## Goal

Add context parallelism without changing model math:

- enable `parallel.mesh.context`;
- configure `parallel.backend_kwargs.cp`;
- bind model attention module metadata through `CP_MODULE_CONFIG`;
- shard sequence inputs, labels, positions, and masks across context ranks with
  `mvp_engine.kit.CPKit.prepare_causal_batch`;
- keep context ranks on the same samples and micro-batches;
- call `mvp_engine.distributed.cp.sync_cp_grads(...)` after gradient rescale and
  before gradient clipping;
- coordinate CP grad sync with TP sync when both touch the same parameter.

The shared runtime contract is fixed:
`mvp_engine/distributed/parallelize.py` calls
`parallelize_model_with_context_parallel(model, get_context_parallel_mesh(device_mesh), cp_config)`
before TP/FSDP2 wrapping when `parallel.mesh.context > 1`. The model must expose
`CP_MODULE_CONFIG`, mapping attention module class names to metadata such as
`{"qkv_layout": "BHSD"}`. When `cp.grad_sync=true`, the runtime attaches
`_cp_grad_sync`; the engine optimizer step must call `sync_cp_grads(model)`.

## Required Inputs

Identify these before editing:

- target recipe path;
- top-level model class used by training;
- attention module classes and Q/K/V layout;
- hidden-state layout and sequence dimension;
- RoPE/position-id/mask handling;
- whether attention uses MHA, GQA, MQA, or packed samples;
- dataset/sampler rank logic and data-parallel mesh dimensions;
- intended `parallel.mesh.context`;
- active TP/FSDP2 mesh settings;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask only if context size, target model, or available devices cannot be derived.

## Workflow

### 1. Locate Attention And Data Sharding

Search the recipe first:

```bash
rg -n "CP_MODULE_CONFIG|parallelize_model|parallel.mesh|backend_kwargs.*cp" recipes/<recipe>
rg -n "q_proj|k_proj|v_proj|o_proj|scaled_dot_product_attention|flash_attn|attention" recipes/<recipe>
rg -n "DistributedSampler|get_world_size|get_rank|get_data_parallel|device_mesh|RuntimeContext|dp_dims" recipes/<recipe>
```

Find the real top-level class, attention modules, and sampler/loader code.

### 2. Configure Mesh And Backend

Use a separate context mesh dimension:

```yaml
parallel:
  mesh:
    shard: <fsdp2 shard size>
    context: <context size>
    tensor: <tp size>
  backend_kwargs:
    cp:
      implementation: ulysses
      attn_implementation: flash_attention_2
      grad_sync: true
```

Rules:

- `context > 1` activates context parallel attention;
- Ulysses degree is inferred from `parallel.mesh.context`;
- `shard > 1` is required because this repo rejects pure model parallelism
  without FSDP2;
- `tp.builtin_sequence_parallel` may be combined with `context > 1`; SP still
  uses the tensor mesh, so verify tensor/context data identity, per-module
  sequence dimensions, and grad ownership together;
- all ranks that differ only by `context` must read the same samples;
- exclude `tensor` and `context` from data-loader sharding and global batch
  accounting;
- include `context` in token/loss statistics reduction.

### 3. Bind Attention Metadata

Bind `CP_MODULE_CONFIG` on the real top-level model class:

```python
class <TopModelClass>(...):
    CP_MODULE_CONFIG = {
        "<AttentionRuntimeClass>": {
            "qkv_layout": "BHSD",
        },
    }
```

Supported layouts are `BSHD` and `BHSD`. The shared CP runtime registers the
`ulysses_sp` attention implementation and sets matched modules to use it.

### 4. Prepare Local Batches

Use `mvp_engine.kit.CPKit` in recipe engines:

```python
prepared = self.cp_kit.prepare_causal_batch(
    batch,
    device_mesh=self.device_mesh,
    pad_token_id=pad_token_id,
)
ctx.data = prepared.local_batch
```

The helper pads to a context-size multiple, globally shifts next-token labels,
extracts token-aligned local tensors, and returns global-position local
`position_ids`.

For packed multimodal batches, select the media-boundary strategy explicitly:

```python
prepared = self.cp_kit.prepare_packed_causal_batch(
    batch,
    device_mesh=self.device_mesh,
    pad_token_id=pad_token_id,
    split_strategy="multimodal",
    global_packed_seq_params_key="global_packed_seq_params",
    temporal_patch_size=temporal_patch_size,
)
ctx.data = prepared.local_batch
```

Rules:

- `split_strategy="text"` is the default token-even split for text-only data;
- `split_strategy="multimodal"` reads media lengths from
  `global_packed_seq_params.cu_seqlens_q`;
- image split points must stay on image boundaries;
- video split points must stay on video boundaries, or tubelet boundaries when
  `temporal_patch_size > 1`;
- `num_frames` metadata is required for videos when `temporal_patch_size > 1`.

### 5. Sync CP Gradients

At synchronized optimizer steps:

```python
stats = self.token_loss_kit.reduce_window()
self.scaler.unscale_(self.optimizer)
self.token_loss_kit.rescale_gradients(self.model.parameters(), stats)
sync_cp_grads(self.model)
clip_grad_norm_(self.model, max_grad_norm)
```

Keep the call after token/global loss rescale and before clipping or
`optimizer.step()`.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- `parallel.mesh.context` and `backend_kwargs.cp` are present;
- if tensor-mesh built-in sequence parallel is enabled, tensor mesh is active
  and the recipe validates CP+SP layout and grad handling together;
- dataloading excludes `context` from sample sharding;
- token/loss statistics include `context`;
- the real top-level class binds `CP_MODULE_CONFIG`;
- engine batch preparation uses `mvp_engine.kit.CPKit.prepare_causal_batch`;
- `optimizer_step()` calls `sync_cp_grads(...)` before clipping;
- attention receives and returns local-sequence tensors;
- RoPE/position/mask/loss boundaries are explicitly handled;
- CP and TP grad hooks do not independently sync the same parameter.

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
  --config-override optim.mixed_precision=bf16
```

Add optional impact validation when correctness depends on attention parity,
multimodal spans, packing boundaries, or gradient accumulation.

## Output

- State which model/config/test files changed.
- Summarize context size.
- State which attention modules are covered by `CP_MODULE_CONFIG`.
- State where `CPKit` prepares batches and where `sync_cp_grads` runs.
- Report structure, smoke, and optional impact validation status.
- Call out missing runtime dependencies such as `flash-attn`.

## Read On Demand

- `references/cp_rules.md`: context mesh, Ulysses, boundary, and validation
  rules.
- `references/asserts.py`: recipe-local assertion template.
