---
name: vlm-data-pipeline
description: Understand, modify, or port Basic VLM-style multimodal data pipelines from raw rows to model batches, including dataset backend choice, schema contracts, preprocessing, guards, processor setup, collation, and optional data preparation.
---

# VLM Data Pipeline

## Goal

This skill documents the Basic VLM data pipeline pattern used by `recipes/basic_vlm`.

Use it in two modes:

- **Basic VLM mode:** If the target recipe is `basic_vlm` or derived from it, do not reimplement the pipeline. Treat this skill as a design map for the existing code and modify only the layer required by the request.
- **Porting mode:** If another VLM recipe does not have a complete multimodal data pipeline, use this skill to adapt the `basic_vlm` design to the target recipe, processor, model, and dataset backend.

Keep implementation recipe-local unless the user explicitly asks for shared engine behavior.

### Inputs

- A target VLM recipe or planned recipe.
- The raw dataset schema, including conversation, image reference, and image-size fields.
- The target model processor and its multimodal placeholder/token conventions.
- The dataset backend, such as `mvp_dataset`, PyTorch `IterableDataset`, Hugging Face `datasets`, or a custom pipeline.

### Outputs

- A documented raw-to-batch data path.
- A normalized sample contract with token tensors, labels, optional multimodal tensors, and optional packed metadata.
- Clear invalid-sample, late-materialization, resume, and accounting behavior.
- Validation notes for the changed or ported pipeline.

### Failure Modes

- Raw schema and processor placeholder conventions disagree.
- Image references are resolved too early or too late for the chosen backend.
- Invalid samples are silently kept, dropped at the wrong boundary, or counted inconsistently.
- Collation produces model-invalid text-only or mixed packed/unpacked batches.
- Non-`mvp_dataset` recipes copy `Assembler`, `RuntimeContext`, `resolve_ref`, or worker-slot assumptions that do not exist.

### Boundaries

- Do not cover packing internals here; use `skills/data/vlm-packing/SKILL.md`.
- Do not cover token-normalized loss, freeze policy, MFU, gradient checkpointing, or model compilation here.
- Do not make the target recipe Qwen3-VL-only. Qwen3-VL is the `basic_vlm` reference model; ported recipes must replace processor-specific chat, image token, resize, and tensor-name rules.
- Do not require `mvp_dataset` for every recipe. Always identify the backend first, then adapt the lifecycle to that backend.

## Existing Reference Implementation

Reference in `basic_vlm`:

- `recipes/basic_vlm/dataset/dataset.py`: runtime dataset chain, transform order, reference resolution, and skip boundary.
- `recipes/basic_vlm/dataset/preprocess.py`: raw schema normalization, image handling, chat rendering, tokenization, and label construction.
- `recipes/basic_vlm/dataset/processor.py`: Qwen3-VL processor loading, tokenizer padding, image pixel limits, and processor fingerprint.
- `recipes/basic_vlm/dataset/collator.py`: token padding, multimodal tensor concatenation, text-only dummy image fallback, and packed metadata padding.
- `recipes/basic_vlm/dataset/types.py`: model-facing batch fields.
- `recipes/basic_vlm/guards/data.py`: staged data guards, empty sentinels, and skip logging.
- `recipes/basic_vlm/dataset/splitter.py`: optional long-conversation splitting before preprocessing.
- `recipes/basic_vlm/tools/build_alignment_data.py`: external data conversion into the recipe schema.
- `recipes/basic_vlm/tools/shuffle_stage2_parquet.py`: optional large-scale parquet reshuffling.

`recipes/basic_vlm/dataset/preprocess_v2.py` is optional advanced reference. The active pipeline in `dataset.py` currently uses `preprocess.py`.

## Workflow

### 1. Identify The Starting Point

Search the target recipe before editing:

```bash
rg -n "build_dataset|process_sample|convert_images_to_pixel_values|DataGuard|build_dataguard|collator|AutoProcessor|apply_chat_template|messages|conversations|image_size|img_size" recipes/<recipe> mvp_engine
```

- If the recipe already follows `basic_vlm`, inspect the existing owner file and make a local change.
- If the recipe has no VLM data pipeline, use `recipes/basic_vlm` as the implementation reference.
- If the request is only about sample packing, use `skills/data/vlm-packing/SKILL.md` instead.
- If packing is enabled while changing this pipeline, keep the boundary with `vlm-packing`: this skill owns raw-to-tokenized-to-collated behavior; `vlm-packing` owns grouping, packed metadata, and packed accounting details.

### 2. Identify Dataset Backend

Before copying a pipeline shape, identify whether the target recipe uses `mvp_dataset`:

```bash
rg -n "mvp_dataset|Dataset\\.from_source|Dataset\\.assemble|RuntimeContext|resolve_ref|TorchLoader|IterableDataset|torch\\.utils\\.data|datasets\\.Dataset|DataPipe" recipes/<recipe> mvp_engine
```

For `mvp_dataset` pipelines, `basic_vlm` uses this shape:

```text
source -> guard -> preprocess/tokenize -> guard -> optional pack/skip marker
       -> resolve refs -> image tensor materialization -> guard -> optional finalize -> collate
```

Reference:

- `recipes/basic_vlm/dataset/dataset.py` uses `Dataset.from_source`, `RuntimeContext`, `assemble`, `map`, and `resolve_ref`.
- `recipes/basic_vlm/engine/basic_vlm_engine.py` consumes the dataset through `mvp_dataset.TorchLoader`.

For non-`mvp_dataset` pipelines, re-decide these behaviors instead of copying `basic_vlm` assumptions:

- iterable lifecycle and where transforms run;
- worker/rank/epoch seed derivation;
- resume and skip boundary;
- image or media reference materialization point;
- whether accounting is based on raw rows, processed samples, packed samples, or batches;
- how errors are filtered, surfaced, or converted to sentinels.

### 3. Raw Schema Contract

`basic_vlm` expects raw rows with:

- `messages` or `conversations`: a list of conversation turns;
- `images`: a list, even for text-only samples;
- `img_size` or `image_size`: one size entry per image, empty for text-only samples;
- optional metadata such as `id`, `source`, `__source__`, `__key__`, or `__global_index__`.

Guidelines:

- Normalize roles before rendering. Common aliases are `human` to `user` and `gpt` to `assistant`; preserve `system` and `tool` only if the target processor/model supports them.
- Treat `<image>` placeholder count and image reference count as a contract.
- Validate image sizes before expensive tokenization or image IO. Sizes must be two positive integers and must align with the image list.
- Keep raw image references as references until the chosen materialization point.
- Document any recipe-specific schema deviations near the preprocessing code.

Reference:

- `recipes/basic_vlm/guards/data.py` owns lightweight raw format and image-size checks.
- `recipes/basic_vlm/dataset/preprocess.py` owns role normalization, image placeholder handling, and image-size normalization.

### 4. Preprocess Layer

The preprocess layer converts one valid raw row into model-facing tensors and lightweight metadata.

Reference in `basic_vlm`:

- `process_image` accepts dictionary refs, bytes, PIL images, and paths, then returns RGB images.
- `process_sample` normalizes messages, validates placeholder/image counts, applies thinking-mode policy, renders chat templates, computes image token estimates, tokenizes, and builds labels.
- `convert_images_to_pixel_values` runs after reference resolution and writes `pixel_values` and `image_grid_thw`.

Guidelines:

- Use the target processor's real chat template and multimodal token conventions.
- Keep assistant-only labels and mask all prompt, system, user, tool, padding, and vision-token positions with the recipe's ignore index.
- Drop or sentinel samples with no supervised assistant tokens.
- Recompute image token estimates whenever image resizing or model processor rules change.
- Preserve placeholder order so image refs, image sizes, `pixel_values`, and `image_grid_thw` remain aligned.
- In `basic_vlm`, late materialization means raw image refs survive tokenization and optional packing; image tensors are created only after `resolve_ref`.

### 5. Guard And Skip Policy

Use staged validation so bad samples are removed before expensive work and late failures do not break the loader.

Reference in `basic_vlm`:

- `recipes/basic_vlm/guards/data.py` defines `DataGuard` and `build_empty_sample`.
- `recipes/basic_vlm/dataset/dataset.py` runs guards before preprocessing, after preprocessing, and after image materialization.

Guidelines:

- Guard raw shape before preprocessing.
- Guard non-empty `input_ids` after preprocessing.
- Guard again after late image decode/materialization, because media failures can create empty sentinels.
- For `mvp_dataset`, use `assemble` for filtering stages that may drop samples or packed groups.
- For non-`mvp_dataset`, define the equivalent filter/drop/sentinel mechanism explicitly.
- Fail fast for schema or processor configuration errors that make all samples invalid; filter per-sample data defects that are expected in large datasets.

### 6. Processor And Model-Specific Rules

The processor is part of the data contract, not only a model-loading detail.

Reference in `basic_vlm`:

- `recipes/basic_vlm/dataset/processor.py` loads `AutoProcessor`, sets tokenizer right padding, fills missing pad token from EOS when needed, applies image pixel limits, and installs a processor fingerprint.
- `recipes/basic_vlm/dataset/preprocess.py` reads image processor geometry such as patch size, merge size, and pixel limits.

Guidelines:

- Do not hard-code Qwen3-VL image token rules into a non-Qwen recipe.
- If the model changes, recheck chat template rendering, special tokens, vision placeholder expansion, image resize rules, and output tensor names.
- Keep processor cache fingerprints stable when the backend caches mapped outputs.

### 7. Collator Layer

The collator turns processed samples into one model batch.

Reference in `basic_vlm`:

- `recipes/basic_vlm/dataset/collator.py` pads `input_ids`, `attention_mask`, and `labels`.
- It concatenates `pixel_values` and `image_grid_thw`.
- It appends one ignored-label dummy multimodal suffix for text-only local batches so Qwen3-VL still receives valid vision tensors.
- It pads `pack_segment_ids` and rejects mixed packed/unpacked batches.

Guidelines:

- Pad `labels` with the ignore index.
- Keep text-only and multimodal samples compatible with the target model's forward path.
- If adding a dummy media payload, ensure its labels are ignored and any packed metadata remains isolated.
- Do not put packing into the collator unless the recipe intentionally makes the collated batch the packing boundary; if so, recheck resume and accounting.

### 8. Optional Data Preparation

Runtime pipeline changes often require matching offline data preparation.

Reference in `basic_vlm`:

- `recipes/basic_vlm/tools/build_alignment_data.py` converts external parquet data into the recipe schema and preserves image refs, placeholder counts, and sample statistics.
- `recipes/basic_vlm/tools/shuffle_stage2_parquet.py` is an optional large-scale parquet reshuffle utility, not a runtime dataset requirement.

Guidelines:

- Keep conversion tools aligned with the raw schema contract.
- Preserve enough metadata to debug dropped samples.
- Validate placeholder/image count before writing expensive output shards.
- Do not require an offline shuffle step unless the dataset source and training stage need it.

## Validation

For data pipeline changes, check at least:

- text-only sample;
- single-image sample;
- multi-image sample;
- placeholder/image count mismatch;
- bad image-size metadata;
- image materialization failure;
- no supervised assistant token;
- collator behavior for a text-only local batch.

If packing is enabled, also run the validation checklist from `skills/data/vlm-packing/SKILL.md`.

## Output

- State whether you modified existing `basic_vlm` behavior or ported the design into another recipe.
- State the dataset backend, raw schema, materialization point, invalid-sample policy, and collator behavior.
- State whether packing is involved and, if so, which parts were delegated to `vlm-packing`.
- State what validation ran and what remains unverified.
