# MLLM DataKit Packing

The standard MLLM DataKit pipeline is always packed.

Use `PackingOptions` for active knobs:

```python
PackingOptions(
    selection_strategy="best_fit",
    open_pack_limit=8,
    buffer_size=64,
)
```

Do not add a config field that disables packing for the standard Basic VLM
style recipe. If a new recipe truly cannot support packing, document why it
does not use the standard MLLM DataKit pipeline.

Packed samples should contain:

- concatenated `input_ids`, `attention_mask`, `labels`;
- `pack_segment_ids` for source-sample boundaries;
- `source_sample_num`;
- media tensors or refs merged in placeholder order;
- token counters produced by the collator.

Model-specific packed attention and position conversion still belongs in the
recipe/model preparation path.
