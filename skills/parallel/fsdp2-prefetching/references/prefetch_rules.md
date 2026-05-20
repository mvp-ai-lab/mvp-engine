# FSDP2 Prefetch Rules

Use this reference when choosing custom FSDP2 prefetch edges for a recipe-local
model hook.

## Runtime Contract

`mvp_engine/distributed/fsdp2.py` applies FSDP2 wrapping first, then resolves:

```python
model.__class__.APPLY_FSDP2_CUSTOM_PREFETCHING(model)
```

The hook must therefore read live module instances from `model` after wrapping
and call PyTorch FSDP2 methods on those modules:

```python
module.set_modules_to_forward_prefetch([...])
module.set_modules_to_backward_prefetch([...])
```

## Edge Rules

- Add edges only between FSDP2-wrapped modules.
- Forward prefetch should point from the currently executing wrapped module to
  the next wrapped module that will execute soon.
- Backward prefetch should follow the reverse dependency order.
- Prefer the minimum useful edge set. Do not connect every possible module pair.
- For branchy models, follow real source execution order and branch joins. Do
  not assume branches execute in parallel unless the model actually does.
- For shared modules, avoid duplicate edges that point to the same module object
  through different names.
- For high-precision FSDP2 module groups, treat them as wrapped modules if
  `fully_shard()` is applied to them.

## Common Acceleration Practices

Custom prefetching helps when the next wrapped module is known early enough that
its unshard/all-gather can overlap with useful compute. It is most useful for
large wrapped blocks and stage handoffs where default traversal does not match
the real forward order.

Common cases:

- Sequential transformer or ViT stacks: prefetch `layer[i + 1]` while executing
  `layer[i]`, and prefetch `layer[i - 1]` during backward. This is the safest
  baseline when all layers are similar size.
- VLM stage transitions: connect the last wrapped vision block to the first
  wrapped projector or language block that actually runs next. This can hide
  communication at vision-to-language handoff boundaries.
- Encoder-decoder or cross-attention models: connect the real transition from
  encoder blocks into decoder or cross-attention blocks, rather than assuming a
  single flat stack.
- Branch joins: prefetch the first wrapped module after the join from the last
  wrapped module that executes before the join. Keep branch-local edges separate
  if branches run in a fixed source order.
- Large head or adapter blocks: add an edge only if the block is FSDP2-wrapped
  and large enough that its all-gather is visible in the profiler.
- High-precision wrapped modules: include them only when they are real wrapped
  compute blocks. Do not prefetch tiny norm layers just because they are in a
  high-precision allowlist.

Tuning rules:

- Start with one-module lookahead. Increase prefetch distance only when profiler
  traces still show waiting and memory headroom is sufficient.
- Use explicit forward prefetch mainly for static execution order, CPU-bound
  training loops, or models whose Python-side glue delays the next FSDP2
  pre-forward hook. For simple GPU-bound LLM stacks, implicit forward prefetch
  is often enough.
- Prefer backward prefetch when backward idle gaps dominate; it often matters
  more than forward prefetch for deep stacks.
- Treat two-module lookahead as a tuning experiment, not a default. It can hide
  longer communication gaps, but it keeps more unsharded parameter buffers live.
- Do not prefetch across optional branches unless the target always runs for the
  current training path.
- Avoid over-prefetching. More edges can raise memory pressure, extend live
  parameter windows, and reduce overlap if too many all-gathers compete.
- Keep `reshard_after_forward` and mixed precision unchanged unless a separate
  measurement justifies changing them. Prefetch edges should be the first tuning
  variable.
- Treat throughput gains as workload-specific. Validate with steady-state step
  time and profiler evidence, not only with a structure test.

## Wrap Granularity

Prefetching quality depends on the FSDP2 wrap plan. A good prefetch edge cannot
fix a poor communication group.

- Prefer repeated compute blocks such as transformer blocks, ViT blocks, decoder
  layers, or large modality-specific blocks as FSDP2 targets.
- Avoid wrapping the whole model as one unit when throughput matters; there is no
  next unit to overlap with current compute.
- Avoid wrapping tiny modules such as individual norms, small projections, or
  short adapters unless there is a specific reason. Too many small groups can
  add collective latency and make prefetch less useful.
- Keep similarly sized repeated layers in the same wrap family when possible.
  Uniform groups make communication and compute easier to pipeline.
- If embeddings or output heads are large and untied, consider whether they need
  their own FSDP2 group. If they stay in the root group, their live interval may
  be much longer than expected.

## Forward Prefetch Caveats

Forward prefetch changes when the next all-gather is issued. It can improve
overlap, but it also increases live parameter memory.

- The forward order must be stable for the training path being optimized.
- Dynamic routing, skipped layers, data-dependent branches, or multimodal
  optional inputs require conservative edges or no forward prefetch across the
  dynamic boundary.
- The first all-gather in a step has nothing before it to overlap with. If this
  gap matters, an explicit early `model.unshard()` can be useful, but only when
  the recipe can safely place it before input preparation or other work.
- If profiler traces already show good overlap from implicit prefetch, leave
  forward prefetch alone and tune backward edges or wrapping instead.

## Profiling Checklist

Before claiming a speedup, compare prefetch off and on with the same config,
global batch size, precision, checkpointing, and warmup policy.

Check:

- steady-state step time after warmup;
- GPU idle gaps around FSDP all-gather and reduce-scatter;
- peak memory and allocation spikes;
- whether the all-gather for the next wrapped module overlaps current compute;
- whether backward all-gather and reduce-scatter serialize unexpectedly;
- whether the first forward block or last backward block remains exposed, which
  is expected and not always fixable.

## Sequential Stack

For a simple stack:

```python
layers = list(model.encoder.layers)
for idx, layer in enumerate(layers):
    if idx + 1 < len(layers):
        layer.set_modules_to_forward_prefetch([layers[idx + 1]])
    if idx > 0:
        layer.set_modules_to_backward_prefetch([layers[idx - 1]])
```

## Branch Transition

For a model that runs `vision -> projector -> language`, connect only the
wrapped handoff points:

```python
vision_last.set_modules_to_forward_prefetch([projector_block])
projector_block.set_modules_to_forward_prefetch([language_first])

language_first.set_modules_to_backward_prefetch([projector_block])
projector_block.set_modules_to_backward_prefetch([vision_last])
```

Skip unwrapped modules in the middle. If the projector is not FSDP2-wrapped,
prefetch from `vision_last` to the next wrapped language block instead.

## Idempotence

Every hook should be safe to call twice:

```python
if getattr(model, "_fsdp2_prefetching_configured", False):
    return
...
model._fsdp2_prefetching_configured = True
```

## Impact Validation

Structure and smoke tests can prove that the hook exists and runs. Edge identity
requires an impact test:

- build the real FSDP2-wrapped model;
- resolve expected module paths to live module objects;
- inspect the installed forward/backward prefetch targets;
- compare by object identity, not by string name.
