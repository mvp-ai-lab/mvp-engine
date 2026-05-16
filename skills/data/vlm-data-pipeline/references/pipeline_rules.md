# VLM Data Pipeline Rules

Use this reference when implementing or reviewing a recipe-local VLM data
pipeline.

## Reference Shape

`recipes/basic_vlm` is the richest local reference:

- `dataset/dataset.py`: dataset backend, transform order, reference resolution,
  and materialization boundary;
- `dataset/preprocess.py`: raw schema normalization, media placeholder handling,
  chat rendering, tokenization, and label construction;
- `dataset/processor.py`: processor loading, padding, pixel limits, and cache
  fingerprinting;
- `dataset/collator.py`: text/media collation and text-only local-batch handling;
- `dataset/types.py`: model-facing batch fields;
- `guards/data.py`: staged raw and processed sample guards.

Use these as patterns, but replace processor-specific details for non-Qwen
recipes.

## Backend Lifecycle

Identify the backend before copying code:

- `mvp_dataset`: supports staged `map`, `assemble`, reference resolution, and
  late materialization.
- PyTorch `Dataset`: usually preprocesses inside `__getitem__` or during dataset
  construction.
- PyTorch `IterableDataset`: must handle worker/rank sharding, seeding, and
  skip accounting explicitly.
- Hugging Face `datasets`: may cache mapped outputs, so processor fingerprints
  and schema versioning matter.

For each backend, define:

- where raw rows are validated;
- where media refs are resolved;
- where image/video bytes become tensors;
- where invalid samples are dropped;
- which counter tracks raw, processed, packed, and batched samples.

## Raw Schema Rules

The raw schema should make placeholder/media alignment unambiguous:

- conversations are a list of turns;
- roles are normalized before rendering;
- image/video refs are ordered lists;
- media-size metadata has one entry per media item;
- placeholder count equals media ref count unless the target processor has a
  documented alternative convention;
- metadata such as `id`, `source`, or row index is preserved when useful for
  debugging.

Validate cheap schema errors before tokenization or media IO.

## Preprocess Rules

Preprocess one valid raw row into model-facing tensors and lightweight metadata.

Rules:

- use the target processor's chat template;
- keep media placeholder order stable;
- estimate or expand media tokens using the target processor's real geometry;
- build labels from assistant targets only;
- mask prompt, user, system, tool, media, pad, and truncated positions with the
  ignore index;
- reject samples with no supervised assistant tokens unless the recipe has a
  documented zero-supervision policy;
- do not silently truncate through an image/video token span.

## Media Materialization

Choose a materialization boundary deliberately:

- Early materialization is simpler and catches media failures sooner.
- Late materialization keeps references cheap through tokenization, filtering,
  and optional packing.
- Packed pipelines usually benefit from late materialization because packing can
  drop or reorder samples before expensive media decode.

After materialization, verify:

- media tensor count matches placeholder count;
- `pixel_values`, `image_grid_thw`, video tensors, or equivalent fields use the
  target model's expected names;
- text-only samples follow the target model's valid text-only path or receive a
  fully ignored dummy media payload.

## Guard And Skip Policy

Use staged validation:

- raw guard: required fields, role shape, media refs, media sizes;
- preprocess guard: non-empty `input_ids`, matching tensor lengths, supervised
  tokens;
- materialization guard: media decode success and tensor metadata alignment;
- collation guard: no mixed packed/unpacked batch unless supported.

Fail fast for global configuration errors. Filter or sentinel per-sample data
defects that are expected in large datasets.

## Processor Rules

The processor is part of the data contract:

- normalize tokenizer pad side and pad token;
- configure pixel or frame limits from recipe config;
- verify chat template preserves the source/target split used for labels;
- record a stable fingerprint when mapped outputs are cached;
- recheck special tokens and media tensor names when changing models.

Do not hard-code Qwen image token or resize rules into a non-Qwen recipe.

## Collator Rules

The collator builds one model batch:

- pad `input_ids` with the tokenizer pad id;
- pad `attention_mask` consistently with the model forward path;
- pad `labels` with the ignore index;
- concatenate media tensors in sample order;
- preserve image/video grid metadata order;
- handle text-only local batches explicitly;
- keep packed metadata isolated from padding tokens.

The collator should not become the packing implementation unless the recipe
explicitly chooses collation as the packing boundary.