# Long-Context Attention Rules

## Runtime Contract

The top-level model class must expose:

```python
APPLY_LONG_CONTEXT_ATTENTION: Callable[[nn.Module, DeviceMesh, dict], None]
```

The shared runtime calls the long-context hook, then applies TP and FSDP2.
Context grad sync is installed after
FSDP2 when `parallel.backend_kwargs.long_context.grad_sync=true`.

## Mesh Rules

- `parallel.mesh.context` is the long-context model-parallel size.
- Ulysses degree is inferred from `parallel.mesh.context`.
- `context` is not data parallel and must not multiply global batch size.
- Exclude both `tensor` and `context` from samplers/loaders that choose dataset
  shards.
- Keep mesh order as `replicate / shard / context / tensor`. PyTorch TP needs
  `tensor` to be innermost.

## Ulysses Tensor Shapes

`UlyssesAttention` expects:

```text
q, k, v: [batch, local_seq, heads, head_dim]
out:     [batch, local_seq, heads, head_dim]
```

The attention wrapper:

1. uses Ulysses all-to-all to trade local sequence for local heads;
2. runs local attention over the expanded sequence;
3. all-to-alls back to local sequence.

## Degree Selection

- `parallel.mesh.context` must divide the relevant head count after TP sharding.
- For GQA/MQA, verify the selected attention kernel supports fewer KV heads.
- Use `attn_impl=fa` for training unless the recipe explicitly targets another
  supported local Ulysses attention type.

## Boundary Rules

- Shard `input_ids`, labels, attention masks, position ids, and RoPE caches with
  the same context layout.
- Build global `position_ids` first, then extract local `position_ids`.
- Build global next-token labels first, then extract local labels.
- Prefer `mvp_engine.kit.CPKit.prepare_causal_batch(...)` in recipe engines. It
  pads, globally shifts labels, extracts token-aligned tensors, and returns
  local global-position ids with one layout contract.
- Prefer `mvp_engine.kit.CPKit.compute_cross_entropy_loss(...)` for local
  backward loss and context-reduced logging stats.
- Normalize loss with the global valid-token count across context ranks.
- Attention masks must match the extracted layout. If the recipe cannot express
  a local mask, restrict smoke/training to unpadded causal batches.
- Multimodal placeholder spans, such as image/video tokens, must be all-or-none
  local after extraction. Select feature tensors in the same order that local
  placeholder tokens appear, or reject/resample spans that cross context ranks.
- Routers/top-k/metrics that require global sequence context need explicit
  gather/reduce logic or must be disabled for smoke validation.
- Generation and KV-cache inference need a separate design; do not assume the
  training hook supports incremental decode.

## CP + Data Packing

- Keep `pack_segment_ids` token-aligned with `input_ids`; pad inactive positions
  with `0`.
- Build packed position ids and attention-isolation metadata on the global
  sequence before CP extraction.
- Pass raw unshifted labels into `CPKit.prepare_causal_batch(...)`; it performs
  global next-token shift and masks cross-segment labels.
- Use segment-isolated causal attention for packed batches. Plain causal masks
  allow cross-sample attention and are incorrect.
- Compute backward loss on local logits with
  `CPKit.compute_cross_entropy_loss(...)`; log global loss as
  `global_loss_sum / global_valid_tokens`.
- Context ranks must read the same packed samples; do not shard packing or data
  loading by the context mesh dimension.
- Packed/unpacked parity should compare input ids, shifted labels, position ids,
  logits/top-1, global loss, and grad norms. Small bf16 drift is acceptable when
  absolute positions or kernel shapes differ.

## Gradient Sync

Every context rank processes a different token slice for the same samples.
Replicated and sharded model parameters therefore need gradient contributions
summed across the context group.

Use the shared runtime `grad_sync=true` path instead of recipe-local all-reduce.
It uses post-accumulate hooks and all-reduces only the newly added grad delta, so
gradient accumulation does not over-count older `.grad` values.

When TP also syncs a parameter, coordinate hooks:

- do not stack independent delta hooks on the same parameter;
- use one combined delta hook over context and tensor groups, or replace one
  hook for the shared params;
- keep hook installation after FSDP2 when hooks target parameters;
- make ownership idempotent and visible in recipe-local assertions.

## Common Failure Cases

- `long_context.enabled=true` with `parallel.mesh.context == 1`.
- context ranks read different samples.
- RoPE positions restart from zero on every context rank.
- recipes compute labels with a local-only shift.
- multimodal placeholder spans are split across context ranks.
- `sequence_parallel=true` and `long_context.enabled=true` are both enabled.
- context grad sync is disabled while parameters see local-sequence activations.
- CP and TP hooks both delta-sync the same parameter independently.
