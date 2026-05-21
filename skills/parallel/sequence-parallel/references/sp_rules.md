# Sequence Parallel Rules

Use this reference when adding recipe-local sequence parallelism that reuses
`mvp_engine/distributed/tp.py`.

## Runtime Contract

The top-level model class must expose the normal tensor-parallel plan:

```python
TP_MODULE_CONFIG: dict[str, object]
```

Sequence parallel is enabled by config:

```yaml
parallel:
  backend_kwargs:
    sequence_parallel: true
```

Optional sequence-parallel plans live on the same top-level model class:

```python
SEQUENCE_PARALLEL_MODULE_CONFIG: dict[str, object]
SEQUENCE_PARALLEL_SEQUENCE_DIM: int = 1
```

`SEQUENCE_PARALLEL_MODULE_CONFIG` maps runtime module class names to direct child
module names with the `"sequence"` style. The runtime merges it with
`TP_MODULE_CONFIG` before calling `parallelize_module(...)`.

## Mesh Rules

- SP size equals TP size.
- `parallel.mesh.tensor` is the TP/SP size.
- `parallel.mesh.tensor > 1` is required.
- `parallel.mesh.shard > 1` is required because this repo rejects pure TP/SP
  without FSDP2.
- Prefer sequence length divisible by TP/SP size
  (`seq_len % parallel.mesh.tensor == 0`).
- If sequence length is not divisible, pad in the dataloader/collator and mask
  padding tokens in loss/metrics, or verify every SP path supports uneven shards.
- Do not add `parallel.mesh.sequence`.
- The product of explicit or inferred mesh dimensions must match world size.

## TP/SP Layout Rules

When `sequence_parallel=false`, TP styles keep the existing behavior:

- `"col"` maps to `ColwiseParallel()`;
- `"row"` maps to `RowwiseParallel()`.

When `sequence_parallel=true`, the same TP styles become sequence-layout aware:

- `"col"` maps to `ColwiseParallel(input_layouts=Shard(sequence_dim))`;
- `"row"` maps to `RowwiseParallel(output_layouts=Shard(sequence_dim))`;
- `"sequence"` maps to `SequenceParallel(sequence_dim=sequence_dim, use_local_output=True)`.

Column-parallel modules annotate local tensor inputs as sequence-sharded, then
redistribute to the replicated layout they need before computing. Row-parallel
modules produce sequence-sharded outputs. Do not assume every op inside a
TP-covered module sees the same local sequence length as its caller.

Use `SEQUENCE_PARALLEL_SEQUENCE_DIM = 1` for `[batch, seq, hidden]` tensors. Use
`0` for `[seq, batch, hidden]` tensors.

## Planning Rules

- Keep projection linears in `TP_MODULE_CONFIG` as `"col"` and `"row"`.
- Put norm, dropout, and other safe elementwise hidden-state modules in
  `SEQUENCE_PARALLEL_MODULE_CONFIG` as `"sequence"`.
- Do not mark modules with tensor kwargs as `"sequence"` unless those kwargs are
  already layout-compatible or handled in recipe code. Runtime hooks prepare the
  first positional tensor input, not arbitrary tensor kwargs.
- Use runtime class names as keys.
- Use direct child module names as plan keys.
- Keep TP and SP attributes on the same top-level model class.
- Do not move TP/SP module plans into YAML unless the user explicitly asks for a
  config-driven mechanism.

Common attention block pattern:

```python
MODEL_TP_MODULE_CONFIG = {
    "DecoderLayer": {
        "q_proj": "col",
        "k_proj": "col",
        "v_proj": "col",
        "o_proj": "row",
    },
}

MODEL_SEQUENCE_PARALLEL_MODULE_CONFIG = {
    "DecoderLayer": {
        "input_layernorm": "sequence",
        "post_attention_layernorm": "sequence",
    },
}
```

Common MLP block pattern:

```python
MODEL_TP_MODULE_CONFIG = {
    "DecoderLayer": {
        "gate_proj": "col",
        "up_proj": "col",
        "down_proj": "row",
    },
}
```

If attention and MLP live on the same runtime block class, merge all direct child
entries under that class.

## Entry And Output Boundaries

Review every boundary where tensors enter or leave the TP/SP-covered repeated
blocks.

Entry boundaries:

- Embedding outputs, vision projector outputs, and packed-batch adapters may
  produce full sequence tensors before the first SP-aware column projection.
- If a module expects replicated full sequence input, do not mark it
  `"sequence"` unless the upstream layout is already sequence-sharded.
- Pick the SP entry boundary deliberately. Prefer sharding tokens or hidden
  states before replicated modules that should process local sequence shards.
- Avoid running tied embeddings on full sequence and the tied LM head on local
  sequence shards; that mixes two gradient meanings for one shared parameter.

Output boundaries:

- Row-parallel outputs are sequence-sharded when SP is enabled.
- Final loss, logits gathering, metrics, routing summaries, generation caches,
  and checkpoint-only debug outputs may need full sequence tensors.
- Add explicit gather logic only in recipe code that truly consumes global
  sequence tensors.
- For routed, packed, or cached flows, review derived lengths such as `top_k`,
  packed segment length, and cache slice length. They need either divisibility by
  TP/SP size or explicit uneven-shard handling.

## Norm, Dropout, And Elementwise Modules

Usually safe for `"sequence"`:

- LayerNorm and RMSNorm over hidden dimension;
- dropout applied independently per element;
- activation functions;
- residual dropout when both residual branches have matching sequence-sharded
  layouts.

Review carefully:

- BatchNorm or modules reducing over sequence/batch dimensions;
- modules that take tensor kwargs, such as `scale`, masks, position ids, or
  cache tensors;
- routers or top-k selectors that choose globally across tokens;
- packed-sequence utilities that compute global token positions;
- logging and loss normalization that count global tokens;
- attention mask construction that assumes full local sequence length.

## Replicated Parameters On Sequence Shards

If a non-TP / replicated module consumes only local sequence shards, each TP rank
computes only its local-token gradient contribution.

Required action:

- wrap the module with a SP-aware style that handles replicated parameter grads,
  or
- explicitly all-reduce those parameter grads across the tensor mesh.

With FSDP2, replicated parameters may become DTensors on the FSDP mesh only
(for example `("replicate", "shard")`, with no `"tensor"` mesh dim). For those
parameters, a normal pre-accumulation grad hook may not fire. Use
`register_post_accumulate_grad_hook`, but all-reduce only the newly added local
gradient delta. Re-reducing the whole accumulated `.grad` on every backward
overcounts during gradient accumulation.

Do not validate this with only one backward. Run at least two backward calls
before `zero_grad`; a hook that all-reduces the whole accumulated `.grad` can
pass one backward and fail accumulation.

Use `SUM`, not average, across the tensor mesh: each sequence-parallel rank owns
a different token slice, so the full replicated-parameter gradient is the sum of
those local contributions.

When FSDP2 is required, prefer installing replicated-parameter grad sync after
FSDP2. Do not apply tensor-mesh all-reduce both before and after FSDP2 for the
same gradient path.

Common examples: embeddings, tied LM heads, final projections, routers, fusion
layers, and custom norms/dropouts not covered by `SequenceParallel`.

## Common Failure Cases

- `sequence_parallel=true` with `parallel.mesh.tensor == 1`.
- `sequence_parallel=true` with `parallel.mesh.shard == 1`.
- Sequence length is not divisible by TP/SP size and no explicit padding/masking
  or uneven-shard handling exists.
- A `"sequence"` plan is bound on a wrapper class that training never
  instantiates.
- `SEQUENCE_PARALLEL_SEQUENCE_DIM` does not match hidden-state layout.
- A `"sequence"` module receives tensor kwargs whose layouts were not prepared.
- A row-parallel output is consumed by code that expects a replicated full
  sequence.
- A router, sampler, metric, or loss uses local sequence shards as if they were
  global tensors.
- A replicated parameter sees only local sequence shards but its gradient is not
  synchronized across the tensor mesh.
- The model shards sequence after embedding, while a tied LM head later consumes
  sequence-local outputs.
- TP postprocessors update head counts but sequence-local code still uses a
  cached global hidden dimension.

## Validation Notes

Structure validation can prove the plan and config exist. Smoke validation proves
the SP-active training path runs. It does not prove global parity or speedup.

Use an optional impact test when correctness depends on a nontrivial boundary:

- compare TP/SP-off and TP/SP-on loss or logits on the same deterministic batch;
- compare TP/SP-off and TP/SP-on gradients for replicated parameters that consume
  sequence-sharded activations;
- inspect DTensor local shapes for sequence-sharded hidden-state parameters or
  activations when the recipe has a stable hook point;
- assert global token counts are gathered or reduced before loss/metric
  normalization.
