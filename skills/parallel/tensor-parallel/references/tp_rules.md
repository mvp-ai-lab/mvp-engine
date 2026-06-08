# Tensor Parallel Rules

Use this reference when choosing recipe-local TP plans for
`mvp_engine/distributed/tp.py`.

## Runtime Contract

The top-level model class must expose:

```python
TP_MODULE_CONFIG: dict[str, object]
```

Optional metadata fixes live in:

```python
TP_MODULE_POSTPROCESSORS: dict[str, Callable]
```

`TP_MODULE_CONFIG` maps runtime module class names to plans. Each plan maps
direct child module names to `"col"` or `"row"`:

```python
{
    "AttentionBlock": {
        "q_proj": "col",
        "k_proj": "col",
        "v_proj": "col",
        "o_proj": "row",
    }
}
```

## Planning Rules

- Use runtime class names, not file names or config names.
- Use direct child names from `module.named_children()`.
- Use `"col"` for input-expansion projections whose output feature dimension is
  partitioned across TP ranks.
- Use `"row"` for output-merge projections whose input feature dimension is
  partitioned across TP ranks.
- Keep the plan at repeated compute block granularity. Do not plan unrelated
  helper modules just because they contain a small linear.
- Do not represent TP plans in YAML unless the user explicitly asks for a
  config-driven mechanism.

## Common Plan Patterns

Attention:

```python
{
    "q_proj": "col",
    "k_proj": "col",
    "v_proj": "col",
    "o_proj": "row",
}
```

Fused QKV attention:

```python
{
    "qkv": "col",
    "proj": "row",
}
```

Dense MLP:

```python
{
    "fc1": "col",
    "fc2": "row",
}
```

Gated MLP / SwiGLU:

```python
{
    "gate_proj": "col",
    "up_proj": "col",
    "down_proj": "row",
}
```

VLM projector:

- shard large MLP projector expansion layers as `"col"`;
- shard the final projection back to language hidden size as `"row"`;
- leave tiny or shape-glue projections unsharded unless profiler evidence says
  they matter.

MoE:

- shard expert MLP internals like normal gated MLPs when experts are dense local
  modules;
- review router inputs carefully because routers often need full hidden states;
- do not shard expert-count or routing metadata without a postprocessor.

## Postprocessor Rules

Add a postprocessor when forward uses cached metadata that changes under TP:

- `num_attention_heads`;
- `num_key_value_heads`;
- `all_head_size`;
- local expert count;
- tensor split sizes;
- cached hidden/intermediate dimensions.

Example:

```python
def _adjust_attention_for_tp(module, tp_mesh) -> None:
    tp_size = tp_mesh.size()
    if tp_size <= 1 or getattr(module, "_tp_adjusted", False):
        return
    if module.num_attention_heads % tp_size != 0:
        raise ValueError("num_attention_heads must be divisible by TP size.")
    module.num_attention_heads //= tp_size
    module.all_head_size = module.num_attention_heads * module.attention_head_size
    module._tp_adjusted = True
```

Keep postprocessors idempotent. Prefer updating derived module fields over
mutating config objects.

## Replicated Parameter Grad Sync

Review trainable params that are not sharded by the TP plan but consume TP-local
activations.

Common cases:

- Q/K layernorm;
- `q_norm` / `k_norm`;
- per-head norm or scale params;
- small adapter params placed after a colwise projection.

Rules:

- If the param remains replicated, all-reduce its grad over the tensor group.
- If CP also syncs that param, use one combined delta hook over both groups, or
  explicitly replace/exclude one hook.
- Do not stack independent post-accumulate delta hooks on the same param.
- Install param hooks after FSDP2 unless you know the wrapper preserves them.
- Add parity metrics for these params; loss/logits can hide bad local grads.

## Mesh Rules

`parallel.mesh.tensor` is the TP size. In this repo, TP is applied before FSDP2
inside `parallelize_model(...)`, and pure TP without FSDP2 is rejected.

Use:

```yaml
parallel:
  mesh:
    replicate: <data-parallel replicas>
    shard: <fsdp2 shard size>
    tensor: <tp size>
```

Rules:

- `tensor > 1` requires `shard > 1`.
- The product of inferred or explicit mesh dimensions must match world size.
- Attention head counts and any grouped-query KV head counts must be divisible
  by TP size unless the recipe has a specialized sharding strategy.
- Preserve global batch semantics when changing `replicate`.

## Data-Loading Rules

Tensor-parallel ranks cooperate on the same logical model replica. They must not
consume different samples.

- All ranks in the same `tensor` group must receive identical samples and
  micro-batches.
- `tensor` is not data-parallel and must not contribute to dataset sharding,
  dataloader slots, or global batch size.
- Shard data over data-parallel dimensions only, normally `replicate` and FSDP2
  `shard`.
- Exclude `tensor` and other non-data-parallel dimensions such as `context` from
  sampler or `RuntimeContext` sharding.
- For `mvp_dataset`, construct `RuntimeContext` with `device_mesh` and `dp_dims`
  that include only data-parallel dimensions, or provide an equivalent
  recipe-local sampler guarantee.
- Validate recipe loaders with a stable sample id or batch fingerprint when
  changing mesh-aware data loading.

## Sharding Impact Validation

Structure checks can prove the plan exists. Smoke checks can prove the TP-active
training path runs. Sharding impact requires inspecting local parameter shapes.

For each TP-covered child linear:

- record the pre-parallel global parameter shape;
- build the TP-active model;
- read DTensor local shapes with `param.to_local().shape` when available;
- for `"col"`, expect output-feature dimension to be divided by TP size;
- for `"row"`, expect input-feature dimension to be divided by TP size before
  any additional FSDP2 local sharding;
- compare by module path and parameter name, not only by class name.

## Loss Parity Impact Validation

Tensor parallelism should preserve model semantics. For the same weights and
same batch, TP-off and TP-on forward loss should match within expected numeric
tolerance.

Use this validation when a TP plan or postprocessor is complex enough that a
shape check is insufficient.

Recommended setup:

- fix random seeds;
- build a TP-off reference model and a TP-on model from identical weights;
- use eval mode or otherwise disable dropout and data augmentation;
- run one deterministic batch without an optimizer step;
- compare scalar loss when the recipe has a stable loss path;
- compare selected logits or hidden states when loss includes extra stochastic
  or distributed reductions;
- use dtype-aware tolerances, stricter for fp32 and looser for bf16/fp16.

Common failure signals:

- missing attention-head or KV-head postprocessor;
- wrong `"col"` versus `"row"` placement;
- sharded router input that expected full hidden states;
- output projection missing the row-parallel reduction;
- replicated TP-local norm grads missing tensor-group sync;
- TP and CP grad hooks both updating the same param independently;
- tensor-group ranks reading different samples or different packed-batch layouts;
- TP-on run using a different precision, dropout state, or packed batch layout.

Loss parity proves semantic preservation for the tested path. It does not prove
that TP improved throughput or memory.
