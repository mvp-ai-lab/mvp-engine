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
- Reduce token/loss statistics across non-`tensor` mesh dimensions. CP combines
  parameter gradients through explicit CP gradient synchronization.
- Use `parallel.backend_kwargs.cp.grad_reduce_dtype=float32` for BF16/FP16
  training to reduce CP gradient summation error. Use `same` only when CP
  gradient sync bandwidth is the bottleneck.
- Keep mesh order as `replicate / shard / context / tensor`. PyTorch TP needs
  `tensor` to be innermost.
- CP is not currently compatible with `tp.builtin_sequence_parallel=true`;
  `parallelize_model` rejects this combination.

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

- Recipe `train_pre_step` should return ready local batches: local
  `input_ids`, labels, shifted labels, `pack_segment_ids`, position ids, and
  model-family local media tensors.
- Build global `position_ids` first, then extract local `position_ids`.
- DataKit must provide global, segment-safe `shift_labels`; CP then extracts
  local labels and local shifted labels together.
- Use the base `mvp_engine.kit.CPKit` for dense sequence padding, rank-local
  slicing with local token-count metadata refresh, sequence gather, and
  sequence/hidden layout transforms.
- Put model-family media semantics in CPKit extensions, such as
  `QwenVLCPKit`; do not add Qwen/VL field names to the base kit.
- Build `CPSequenceSpec` lists in the recipe so batch ownership is explicit.
  Model structure values such as Qwen-VL `spatial_merge_size` should stay in
  the recipe/model patch and be passed as `pad_scale=spatial_merge_size**2`.
- Avoid sequence holes. Multimodal model patches may temporarily gather local
  embeddings into full-sequence / hidden-sharded layout for visual merge, then
  scatter back to local-sequence / full-hidden layout before the LLM.
- Reduce token/loss stats across the unique token-owner group, keep gradient
  scale based on the data-parallel average world size, and combine CP parameter
  gradients explicitly before clipping.
- Attention masks must match the extracted layout. If the recipe cannot express
  a local mask, restrict smoke/training to supported causal batches.
- Multimodal placeholder spans may cross context ranks only after the visual
  features have been merged into dense embeddings and the LLM sequence is sliced.
- Routers/top-k/metrics that require global sequence context need explicit
  gather/reduce logic or must be disabled for smoke validation.
- Generation and KV-cache inference need a separate design; do not assume the
  training path supports incremental decode.

## CP + Data Packing

- Keep `pack_segment_ids` token-aligned with `input_ids`; pad inactive positions
  with `0`.
- Build packed position ids and attention-isolation metadata on the global
  sequence before CP extraction.
- Build next-token `shift_labels` in DataKit on the global dense sequence before
  slicing them with `CPKit.slice_sequence_batch(...)`.
- Build packed model metadata before slicing, then keep global topology metadata
  such as `cu_seq_lens_*` while slicing dense token-aligned tensors.
- Prefer packed `cu_seq_lens_*` metadata for Qwen-VL CP. Do not carry prebuilt
  multi-dimensional attention masks into `CPKit.slice_sequence_batch(...)`.
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
- `tp.builtin_sequence_parallel=true` is enabled while `parallel.mesh.context`
  is active.
- `_cp_grad_sync` is attached but `sync_cp_grads(model)` is never called.
- CP and TP sync paths both update the same parameter independently.
