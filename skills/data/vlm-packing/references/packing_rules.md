# VLM Packing Rules

Use this reference when implementing or reviewing VLM sample packing around the
standard MLLM data kit.

## Reference Shape

Primary files:

- `mvp_engine/kit/mllm/data/spec.py`: `MLLMPackingSpec`;
- `mvp_engine/kit/mllm/data/packing.py`: `MLLMPackingAssembler` and
  `build_packed_block_causal_mask`;
- `mvp_engine/kit/mllm/data/sample.py`: `MLLMPack.to_model_inputs`;
- `mvp_engine/kit/mllm/data/media.py`: packed media merge and batch collation hooks;
- `mvp_engine/kit/mllm/utils/step_estimation.py`: packed total-step estimation.

Recipe examples:

- `recipes/qwen3_vl/engine/qwen3_vl_engine.py`;
- `recipes/openbee/engine/openbee_engine.py`;
- recipe-local `model/packing/` modules for packed attention and position logic.

## Packer Rules

The DataKit packer should:

- use tokenized length, not raw text length;
- respect `MLLMPackingSpec.max_seq_len`;
- be deterministic from mvp-dataset assembler context and spec settings;
- bound memory with `buffer_size` and `open_pack_limit`;
- preserve source-sample order inside finalized packs;
- emit `MLLMPack` objects, not final model dictionaries.

For custom algorithms, implement an assembler with the same mvp-dataset
`push`/`finish` contract and pass it via `MLLMPackingSpec.assembler_cls`.

## Final Model-Input Rules

`MLLMPack.to_model_inputs()` should:

- call `sample.load_media()` after refs have been resolved;
- skip samples marked empty by tokenization or media loading;
- concatenate `input_ids`, `attention_mask`, and `labels`;
- create `pack_segment_ids` with segment ids starting at `1`;
- record `source_sample_num`;
- merge media fields in source-sample order.

## Collator Rules

The collator should:

- pad token fields normally;
- pad `pack_segment_ids` with inactive `0`;
- reject or guard invalid text-only backend cases explicitly;
- keep dummy media payload labels ignored and segment-isolated;
- keep media tensor order aligned with placeholders.

Packing is completed before collation.

## Model Preparation Rules

Packed metadata must be converted before model forward when the model/backend
requires it:

- block-causal masks for eager/SDPA paths;
- segment-id masks or cu-seqlens for FlashAttention paths;
- packed multimodal position ids for VLM RoPE rules;
- backend-specific patches when the model library discards custom metadata.

Check that:

- source segments are attention-isolated from each other;
- causal order still holds inside each segment;
- position ids match the model's expected text/media convention;
- image/video grid rows are consumed in placeholder order.

## Accounting Rules

Packing changes the consumption unit:

- step estimation counts packed outputs, not raw rows;
- source-sample counts come from `source_sample_num`;
- token counts should use actual packed tokens after padding/masking;
- effective tokens should come from supervised labels after shifting when
  applicable;
- late media failures preserve packed-output accounting consistency.

Require explicit `loop.total_steps` if the backend cannot provide a reliable
finite packed stream for estimation.
