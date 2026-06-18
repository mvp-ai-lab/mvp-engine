# MLLM DataKit Packing

The standard MLLM DataKit pipeline is packed. `MLLMPackingAssembler` groups
tokenized `MLLMSample` objects and emits `MLLMPack` objects.
`MLLMPack.to_model_inputs()` builds the final packed dictionary.

## Spec Fields

Use `MLLMPackingSpec`:

```python
MLLMPackingSpec(
    max_seq_len=int(config.data.max_seq_len),
    algorithm="multi_pack",
    selection_strategy="best_fit",
    open_pack_limit=8,
    buffer_size=64,
    block_causal=True,
)
```

Fields:

- `max_seq_len`: maximum packed token length.
- `algorithm`: logical algorithm name for logging/config clarity.
- `selection_strategy`: `"best_fit"` for tighter packing or `"random"` for
  randomized placement.
- `open_pack_limit`: maximum number of in-flight packs.
- `buffer_size`: pending sample pool size before draining packs.
- `block_causal`: marks that packed boundaries should be isolated downstream.
- `assembler_cls`: optional custom mvp-dataset assembler.

## Assembler Contract

Use `assembler_cls` for a packing algorithm replacement:

```python
MLLMPackingSpec(..., assembler_cls=MyPackingAssembler)
```

The custom assembler follows the mvp-dataset contract:

- `push(sample: MLLMSample) -> Iterable[MLLMPack]`
- `finish(drop_last: bool = False) -> Iterable[MLLMPack]`
- `state_dict()`, `load_state_dict(...)`, and `fingerprint()` when resumability matters

The assembler groups samples. Final tensor construction belongs to
`MLLMPack.to_model_inputs()`.

## Packed Model Inputs

Final packed dictionaries contain:

- concatenated `input_ids`, `attention_mask`, and `labels`;
- `pack_segment_ids`, with active segment ids starting at `1`;
- `source_sample_num`, the number of source samples represented by the pack;
- media tensors merged by `MLLMMediaHandler.merge_pack(...)`;
- token counters added by the collator.

`pack_segment_ids` is padded later with inactive value `0`. It is the generic
boundary signal for block-causal masks, FlashAttention metadata, packed position
ids, and debugging.

Use `build_packed_block_causal_mask(pack_segment_ids, dtype=...)` for generic
eager/SDPA block-causal masks when the recipe needs it.

## Boundaries

DataKit owns:

- token-length based grouping;
- pack metadata;
- source-sample counting;
- media merge across samples in a pack.

Recipe/model code owns:

- packed attention masks for a specific backend;
- multimodal position ids;
- FlashAttention/cu-seqlens metadata;
- model-specific packed input preparation;
- throughput and loss-token accounting conventions.

Standard MLLM recipes expose packing through the knobs consumed by
`MLLMPackingSpec`.

## Step Estimation

`MLLMStepEstimationKit` consumes packed dataset outputs. It counts packed
samples and reads `source_sample_num` to estimate the packed/source compression
ratio. Use a finite estimation source spec with `resample=False` and usually
`resolve_refs=False`.

Step estimation should use the same packing spec as training so the estimated
packed-output count reflects actual training packing behavior.

## Implementation Checklist

- Packing length uses `sample.token_length`.
- Overlength samples have an explicit policy.
- `finish(drop_last=...)` behavior is understood for finite streams.
- Packed outputs include `pack_segment_ids` and `source_sample_num`.
- Collation pads `pack_segment_ids` with inactive `0`.
- Model preparation isolates attention across packed source segments.
- Step estimation counts packed outputs, not raw rows.
