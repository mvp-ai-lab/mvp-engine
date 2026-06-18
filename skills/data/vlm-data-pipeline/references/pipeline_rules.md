# VLM Data Pipeline Rules

Use this reference when implementing or reviewing a VLM data pipeline that uses
or extends the current `MLLMDataKit`.

## Reference Shape

Primary kit files:

- `mvp_engine/kit/mllm/data/data.py`: processor, dataset, collation, dataloader,
  and device-transfer orchestration;
- `mvp_engine/kit/mllm/data/spec.py`: source, sample, packing, loader, and
  distribution specs;
- `mvp_engine/kit/mllm/data/sample.py`: sample object, lazy tokenization, media
  loading, and packed model-input conversion;
- `mvp_engine/kit/mllm/data/schema.py`: schema-handler base class;
- `mvp_engine/kit/mllm/data/media.py`: media handler base classes;
- `mvp_engine/kit/mllm/data/tokenization.py`: tokenization handler;
- `mvp_engine/kit/mllm/data/packing.py`: packing assembler and block-causal mask helper;
- `mvp_engine/kit/mllm/data/guard.py`: raw, sample, model-input, and text-only guards;
- `mvp_engine/kit/mllm/data/qwen/`: Qwen chat schema and image media handlers;
- `mvp_engine/kit/mllm/utils/step_estimation.py`: packed MLLM step estimation.

Use `recipes/qwen3_vl/engine/qwen3_vl_engine.py` and
`recipes/openbee/engine/openbee_engine.py` as current wiring examples.

## Backend Lifecycle

For `mvp_dataset`, the standard lifecycle is:

```text
from_source -> assemble guard -> map sample -> assemble guard -> assemble packer
-> resolve_ref -> map pack.to_model_inputs -> assemble guard -> TorchLoader.batch
```

For other backends, explicitly define equivalent behavior:

- raw validation;
- rank/worker sharding and seeding;
- media reference resolution;
- media decoding/materialization;
- invalid-sample skip policy;
- packed-output counting;
- batch collation.

## Raw Schema Rules

Raw schema should make placeholder/media alignment unambiguous:

- conversations are ordered turn lists;
- roles are normalized before rendering;
- media refs are ordered fields or explicit media entries;
- size metadata exists for media types that need token-count estimation;
- placeholder count equals media slot count unless a model documents another rule;
- cheap schema errors are rejected before tokenization or media IO.

## Segment And Label Rules

Schema handlers emit `MLLMSegment(type, loss, value)`.

- `type="text"` stores literal text.
- media segment values are media ids that match `MLLMMediaSlot.media_id`.
- `loss=True` marks exactly the spans that should produce supervised labels.
- media segments should normally be `loss=False`.
- prompt, user, system, tool, pad, dummy media, and dropped/truncated positions
  use ignored labels.

Label policy belongs in schema handlers.

## Media Rules

Media handlers should:

- render placeholders before tokenization;
- load heavy values only after refs are resolved;
- return model field names expected by the target model;
- merge fields in pack order;
- collate fields in batch order;
- use an empty-sample sentinel only when the sample should be dropped.

Text-only batches that are invalid for a model backend should be handled by an
explicit batch guard, not by hiding special cases inside the generic collator.

## Processor Rules

The processor is part of the data contract:

- normalize tokenizer pad side and pad token;
- configure image/frame limits from recipe config;
- verify chat templates preserve the source/target split used for labels;
- avoid hard-coding Qwen tokens or resize rules in non-Qwen handlers.

## Collator Rules

The standard collator:

- pads `input_ids` with tokenizer pad id;
- pads `attention_mask` with `0`;
- pads `labels` with ignore index;
- pads `pack_segment_ids` with inactive `0`;
- creates source/sample/token counters;
- delegates media collation to `MLLMMediaHandler.collate`.

Packing and model-specific attention preparation stay outside the collator.
