# VLM Packing Rules

Use this reference when implementing or reviewing recipe-local VLM sample
packing.

## Reference Shape

`recipes/basic_vlm` is the main local reference:

- `configs/schema.py`: packing knobs and validators;
- `configs/stage*.yaml`: active packing settings;
- `dataset/dataset.py`: transform order and materialization placement;
- `dataset/packing.py`: grouping, finalization, segment metadata;
- `dataset/collator.py`: packed metadata padding and mixed-batch rejection;
- `model/packing/`: packed model-input preparation and attention backend patches;
- `engine/basic_vlm_engine.py`: packed preparation, token accounting, logging;
- `utils/misc.py`: packed total-step inference.

Use these files as patterns, but adapt model-specific attention and position
logic for non-Qwen recipes.

## Lifecycle Rules

Packing combines tokenized samples, so it should normally run after preprocess
has produced `input_ids`, `attention_mask`, and `labels`.

Choose placement deliberately:

- pack before media materialization when media IO is expensive;
- finalize after media materialization if the final packed sample needs tensor
  payloads;
- avoid packing inside the collator unless the recipe intentionally makes the
  collated batch the packing boundary;
- define flush, finish, and `drop_last` behavior for the active backend.

For non-`mvp_dataset` backends, do not assume `Assembler`, `RuntimeContext`,
`resolve_ref`, worker slots, or `TorchLoader` exist. Define equivalent lifecycle
and seed behavior explicitly.

## Packer Rules

The packer owns grouping. It should:

- use tokenized length, not raw text length;
- respect `max_seq_len`;
- treat overlength samples according to explicit recipe policy;
- make random packing deterministic from recipe seed, worker id, rank, and epoch
  where available;
- bound memory with `packing_buffer_size` and `packing_open_pack_limit`;
- preserve source order inside a finalized pack.

Common strategies:

- `best_fit`: better length utilization, more deterministic ordering pressure;
- `random`: simpler distribution, often lower utilization;
- standalone overlength: keeps rare long samples without corrupting other packs.

## Finalizer Rules

The finalizer turns a sample group into one packed sample:

- concatenate `input_ids`, `attention_mask`, and `labels`;
- create segment metadata such as `pack_segment_ids`;
- merge `pixel_values`, `image_grid_thw`, video tensors, refs, or metadata in
  source-sample order;
- record `source_sample_num`;
- drop raw text and large unused fields;
- ensure labels never supervise padding or dummy media tokens.

Boundary metadata should use inactive padding values that cannot be confused
with a real segment.

## Collator Rules

The collator should:

- pad packed token fields normally;
- pad packed boundary metadata with an inactive value such as `0`;
- reject mixed packed/unpacked batches unless explicitly supported;
- keep text-only and multimodal packed samples model-valid;
- keep dummy media payload labels ignored and segment-isolated.

Do not use the collator as a hidden packer without documenting the changed
training unit and accounting implications.

## Model Preparation Rules

Packed metadata must be converted before model forward:

- block causal masks for eager/SDPA paths;
- packed segment-id masks or cu-seqlens for FlashAttention paths;
- packed multimodal position ids for VLM RoPE rules;
- backend-specific mask or position patches when the model library discards
  custom metadata.

Check:

- no source segment attends to another source segment;
- position ids restart or offset exactly as the model expects;
- image/video grid metadata is consumed in the same order as placeholders;
- packed text-only and multimodal segments can coexist;
- every attention implementation used by the config receives compatible packed
  metadata.

## Accounting Rules

Packing changes the unit of consumption:

- step inference should count packed outputs, not raw rows;
- token counts should use actual packed tokens after padding/masking rules;
- effective tokens should come from supervised labels after shifting when
  applicable;
- throughput and MFU should use the same token convention as the training loss;
- late media failures must not silently desynchronize packed-output counts.

When the backend cannot infer packed total steps robustly, document that
limitation and require explicit `loop.total_steps`.

## Attention Isolation Impact Validation

Use this optional impact test when changing model preparation, attention masks,
position ids, or backend patches.

Prove that:

- segment A cannot attend to segment B;
- segment B cannot attend to segment A;
- causal order still holds inside each segment;
- position ids are valid for text and media spans;
- image/video grid rows are consumed by the intended source segment.

This validation proves packed-boundary semantics. It does not prove throughput
or model quality.
