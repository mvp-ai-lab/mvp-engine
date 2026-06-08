# Long-Context Attention Rules

## Runtime Contract

The top-level model class must expose:

```python
APPLY_LONG_CONTEXT_ATTENTION: Callable[[nn.Module, DeviceMesh, dict], None]
```

The shared runtime initializes yunchang process groups first, then calls the
hook, then applies TP and FSDP2. Context grad sync is installed after FSDP2 when
`parallel.backend_kwargs.long_context.grad_sync=true`.

## Mesh Rules

- `parallel.mesh.context` is the long-context model-parallel size.
- `context == ulysses_degree * ring_degree`.
- `context` is not data parallel and must not multiply global batch size.
- Exclude both `tensor` and `context` from samplers/loaders that choose dataset
  shards.
- Keep mesh order as `replicate / shard / context / tensor`. PyTorch TP needs
  `tensor` to be innermost; when `tensor=1`, context ranks remain contiguous.
- For 8 nodes x 8 GPUs, prefer `ulysses_degree=8`, `ring_degree=8`,
  `use_ulysses_low=true` when one context group spans all 64 GPUs. This keeps
  Ulysses all-to-all intra-node and Ring communication cross-node.

## Yunchang Tensor Shapes

`LongContextAttention` expects:

```text
q, k, v: [batch, local_seq, heads, head_dim]
out:     [batch, local_seq, heads, head_dim]
```

The attention wrapper:

1. uses Ulysses all-to-all to trade local sequence for local heads;
2. runs Ring attention across KV blocks;
3. all-to-alls back to local sequence.

Use `LongContextAttentionQKVPacked` only when the recipe naturally produces
`[batch, local_seq, 3, heads, head_dim]`.

## Degree Selection

- `ulysses_degree` should divide the relevant head count after TP sharding.
- For GQA/MQA, verify the selected attention kernel supports fewer KV heads.
- Increase `ring_degree` when heads are too few or sequence length dominates.
- Prefer `ring_impl_type=zigzag` for new training recipes. It balances causal
  Ring attention by pairing early and late sequence chunks on each ring rank.
- Use `basic` only for debugging, compatibility, or recipes that need simple
  contiguous rank-local positions.
- Use `attn_impl=fa` for training; yunchang PyTorch ring variants are useful for
  forward checks but do not provide a complete backward path.

## Ring Layout Rules

- `basic` owns contiguous global positions:
  `rank r -> [r * local_seq, (r + 1) * local_seq)`.
- `zigzag` first chunks the global sequence into `2 * ring_degree` pieces. Ring
  rank `r` owns chunk `r` plus mirrored chunk `2 * ring_degree - r - 1`, then
  Ulysses ranks split that local pair. Local positions are therefore
  non-contiguous and may be non-monotonic.
- Use the shared `extract_local_sequence(...)` helper for every token-aligned
  tensor in non-basic layouts.
- Prefer `mvp_engine.kit.CPKit.prepare_causal_batch(...)` in recipe
  engines. It pads, globally shifts labels, extracts token-aligned tensors, and
  returns local global-position ids with one layout contract.
- For `zigzag`, pad the global sequence to a multiple of
  `2 * ring_degree * ulysses_degree` before extraction. Padding labels must be
  `-100`; padding tokens may stay visible to causal attention when they are only
  appended at the end.
- Do not use `get_basic_sequence_offset(...)` for `zigzag`; it is only valid for
  contiguous `basic` slices.

## Boundary Rules

- Shard `input_ids`, labels, attention masks, position ids, and RoPE caches with
  the same context layout.
- Build global `position_ids` first, then extract local `position_ids` with the
  same layout as `input_ids`. This preserves RoPE positions for `zigzag`.
- Build global next-token labels first, then extract local labels with the same
  layout. A local-only causal shift is wrong for `zigzag`.
- Prefer `mvp_engine.kit.CPKit.compute_cross_entropy_loss(...)` for local
  backward loss and context-reduced logging stats.
- Normalize loss with the global valid-token count across context ranks. If the
  logged scalar should reflect the full sample, reduce the loss sum across the
  same context group before data-parallel logging.
- Attention masks must match the extracted layout. If the recipe cannot express
  a non-contiguous mask, restrict smoke/training to unpadded causal batches.
- Packed samples need segment-aware local slicing and loss masking; do not use a
  simple contiguous token split unless every packed segment boundary is handled.
- Multimodal placeholder spans, such as image/video tokens, must be all-or-none
  local after extraction. Select feature tensors in the same order that local
  placeholder tokens appear, or reject/resample spans that cross context ranks.
- Routers/top-k/metrics that require global sequence context need explicit
  gather/reduce logic or must be disabled for smoke validation.
- Generation and KV-cache inference need a separate design; do not assume the
  training hook supports incremental decode.

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
- `context != ulysses_degree * ring_degree`.
- context ranks read different samples.
- RoPE positions restart from zero on every context rank.
- `zigzag` recipes compute labels with a local-only shift.
- `zigzag` recipes use `get_basic_sequence_offset` or assume contiguous slices.
- multimodal placeholder spans are split across context ranks.
- PyTorch yunchang attention implementation is used for a training backward.
- `sequence_parallel=true` and `long_context.enabled=true` are both enabled.
- context grad sync is disabled while parameters see local-sequence activations.
- CP and TP hooks both delta-sync the same parameter independently.
