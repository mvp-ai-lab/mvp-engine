# Context Parallel Runtime Contract

Use this file for stable CP runtime and kit facts that apply before any
model-family or mechanism-specific reference.

This file owns:

- mesh semantics;
- runtime metadata consumed by `parallelize_model(...)`;
- generic `CPKit` batch and layout boundaries;
- the required CP gradient-sync primitive.

This file does not own:

- packed/varlen topology details;
- model-family media ownership;
- custom attention wrapper details;
- auxiliary hidden layout;
- parity test design.

Those belong in the mechanism references.

## Mesh Contract

- `parallel.mesh.context` is the context-parallel model-parallel size.
- `parallel.mesh.context > 1` activates context-parallel attention.
- Ulysses degree is inferred from `parallel.mesh.context`.
- `context` is not data parallel and must not multiply global batch size.
- Exclude both `tensor` and `context` from samplers and loaders that choose
  dataset shards.
- Keep mesh order as `replicate / shard / context / tensor`; PyTorch TP needs
  `tensor` to be innermost.
- CP is not compatible with `tp.builtin_sequence_parallel=true`.
- Use `parallel.backend_kwargs.cp.grad_reduce_dtype=float32` for BF16/FP16
  training unless bandwidth forces `same`.

## Attention Runtime Contract

The top-level model class instantiated by training exposes:

```python
CP_MODULE_CONFIG = {
    "<AttentionRuntimeClass>": {"qkv_layout": "BHSD"},
}
```

`parallelize_model(...)` reads this map when CP is active and routes matching
attention modules through Ulysses.

Supported Q/K/V layouts are `BSHD` and `BHSD`.

`UlyssesSPAttention` consumes:

```text
q, k, v: [batch, local_seq, heads, head_dim]
out:     [batch, local_seq, heads, head_dim]
```

If a model wraps or bypasses the runtime attention module, use
`custom_attention_dispatch.md`.

## CPKit Boundary

Recipe `train_pre_step` returns ready context-local dense batches:

- local token tensors;
- local labels and shifted labels;
- local position ids;
- local model-family dense fields when those fields are sliced;
- topology metadata in the layout expected by the downstream attention path.

Use `mvp_engine.kit.CPKit` for generic operations:

- dense sequence padding;
- rank-local sequence slicing;
- sequence gather;
- sequence/full-hidden layout transforms.

Use recipe-local or model-family helpers only for semantics not owned by the
generic kit, such as media row ownership or model-specific local indices.

## Label And Position Contract

- Build next-token labels on the global dense sequence before CP slicing.
- Slice labels and shifted labels with the same ownership as `input_ids`.
- Build global position ids before extracting context-local positions.
- Do not restart RoPE or absolute positions from zero on each context rank.

## Gradient Sync Contract

Every context rank processes a different token slice for the same logical
samples. Replicated parameters therefore need CP gradient contributions summed
across the context group.

Use:

```python
sync_cp_grads(model)
```

after unscale and token/global gradient rescale, before clipping and
`optimizer.step()`.

When tensor parallelism also syncs a parameter, coordinate ownership so the same
parameter is not independently synchronized twice.
