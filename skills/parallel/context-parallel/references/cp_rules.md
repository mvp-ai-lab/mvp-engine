# Context Parallel Rules

## Runtime Contract

The top-level model class must expose:

```python
CP_MODULE_CONFIG: dict[str, dict[str, str]]
```

The shared runtime reads that map when `parallel.mesh.context > 1`, registers the
`ulysses_sp` attention implementation, and switches matching attention modules
to use it. The runtime also attaches a `_cp_grad_sync` object when
`parallel.backend_kwargs.cp.grad_sync=true`; engines must call
`sync_cp_grads(model)` at synchronized optimizer steps.

## Mesh Rules

- `parallel.mesh.context` is the context-parallel model-parallel size.
- `parallel.mesh.context > 1` activates context-parallel attention.
- Ulysses degree is inferred from `parallel.mesh.context`.
- `context` is not data parallel and must not multiply global batch size.
- Exclude both `tensor` and `context` from samplers/loaders that choose dataset
  shards.
- Include `context` in token/loss statistics reductions for token-normalized
  training.
- Keep mesh order as `replicate / shard / context / tensor`. PyTorch TP needs
  `tensor` to be innermost.

## Ulysses Tensor Shapes

`UlyssesSPAttention` expects:

```text
q, k, v: [batch, local_seq, heads, head_dim]
out:     [batch, local_seq, heads, head_dim]
```

For Hugging Face attention modules that pass `BHSD`, set:

```python
CP_MODULE_CONFIG = {
    "<AttentionRuntimeClass>": {"qkv_layout": "BHSD"},
}
```

Supported QKV layouts are `BSHD` and `BHSD`.

## Boundary Rules

- Shard `input_ids`, labels, attention masks, position ids, and RoPE caches with
  the same context layout.
- Build global `position_ids` first, then extract local `position_ids`.
- Build global next-token labels first, then extract local labels.
- Prefer `mvp_engine.kit.CPKit.prepare_causal_batch(...)` in recipe engines. It
  pads, globally shifts labels, extracts token-aligned tensors, and returns
  local global-position ids with one layout contract.
- Normalize token loss with token/loss statistics reduced across context ranks.
- Attention masks must match the extracted layout. If the recipe cannot express
  a local mask, restrict smoke/training to supported causal batches.
- Multimodal placeholder spans, such as image/video tokens, must be all-or-none
  local after extraction. Select feature tensors in the same order that local
  placeholder tokens appear, or reject/resample spans that cross context ranks.
- Routers/top-k/metrics that require global sequence context need explicit
  gather/reduce logic or must be disabled for smoke validation.
- Generation and KV-cache inference need a separate design; do not assume the
  training path supports incremental decode.

## CP + Data Packing

- Keep `pack_segment_ids` token-aligned with `input_ids`; pad inactive positions
  with `0`.
- Build packed position ids and attention-isolation metadata on the global
  sequence before CP extraction.
- Pass raw unshifted labels into `CPKit.prepare_causal_batch(...)`; it performs
  global next-token shift and masks cross-segment labels.
- Use segment-isolated causal attention for packed batches. Plain causal masks
  allow cross-sample attention and are incorrect.
- Context ranks must read the same packed samples; do not shard packing or data
  loading by the context mesh dimension.
- Packed/unpacked parity should compare input ids, shifted labels, position ids,
  logits/top-1, global loss, and grad norms. Small bf16 drift is acceptable when
  absolute positions or kernel shapes differ.

## Gradient Sync

Every context rank processes a different token slice for the same samples.
Replicated model parameters therefore need gradient contributions summed across
the context group.

Use:

```python
sync_cp_grads(model)
```

after unscale and token/global gradient rescale, before clipping and
`optimizer.step()`.

When TP also syncs a parameter, coordinate ownership:

- do not stack independent sync paths on the same parameter;
- use one combined sync path over context and tensor groups, or explicitly
  exclude one path for shared params;
- make ownership visible in recipe-local assertions.

## Common Failure Cases

- `parallel.mesh.context > 1` on a model without `CP_MODULE_CONFIG`.
- context ranks read different samples.
- RoPE positions restart from zero on every context rank.
- recipes compute labels with a local-only shift.
- multimodal placeholder spans are split across context ranks.
- `tp.builtin_sequence_parallel=true` and `parallel.mesh.context > 1` are both
  enabled.
- `_cp_grad_sync` is attached but `sync_cp_grads(model)` is never called.
- CP and TP sync paths both update the same parameter independently.
