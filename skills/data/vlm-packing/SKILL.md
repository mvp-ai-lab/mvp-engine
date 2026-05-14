---
name: vlm-packing
description: Understand, modify, or port recipe-local sample packing for VLM recipes. Use when working with the existing Basic VLM packing implementation, adapting it for a derived recipe, or adding similar packing support to a recipe that does not yet have it.
---

# VLM Packing

## Goal

This skill documents the recipe-local VLM packing design used by `recipes/basic_vlm`.

Use it in two modes:

- **Basic VLM mode:** If the target recipe is `basic_vlm` or derived from it, do not reimplement packing. Treat this skill as a design map for the existing code and modify only the layer required by the user request.
- **Porting mode:** If the target recipe starts from `minimal_vlm` or another recipe without packing, use this skill to add packing by copying and adapting the `basic_vlm` design.

Packing is recipe-specific. Keep it under `recipes/<recipe>/`; do not move it into `mvp_engine/` unless the user explicitly asks.

### Existing Reference Implementation

The canonical implementation is in `recipes/basic_vlm`:

- `configs/schema.py`: packing config knobs.
- `configs/stage*.yaml`: active packing settings.
- `dataset/dataset.py`: dataset transform order and skip modes.
- `dataset/packing.py`: grouping, finalization, segment metadata.
- `dataset/collator.py`: padding and mixed packed/unpacked handling.
- `dataset/skip.py`: post-pack fast-resume helpers.
- `model/packing/`: packed model-input preparation and attention backend patches.
- `engine/basic_vlm_engine.py`: step inference, fast resume, packed input preparation, token accounting.

When changing a packing detail in `basic_vlm`, first identify which of these files owns that behavior. Avoid broad edits across all layers unless the requested behavior crosses layer boundaries.

### Design Summary

Packing combines multiple already-tokenized samples into one longer model input while preserving source-sample boundaries. A packed sample must:

- concatenate token-aligned fields in source order;
- preserve labels and loss masking;
- merge multimodal payloads in placeholder order;
- carry boundary metadata, such as `position_ids` inputs or segment ids;
- prevent cross-source attention or position leakage;
- keep resume, step inference, token counts, and logging aligned with the packed output unit.

The design has five layers:

1. Config exposes only the knobs the recipe needs.
2. Dataset pipeline chooses where grouping, skip markers, payload materialization, and finalization happen.
3. Packer/finalizer owns sample grouping, concatenation, and boundary metadata.
4. Collator pads packed metadata and rejects unsupported mixed batches.
5. Model preparation converts boundary metadata into model-specific masks, positions, or backend metadata.

## Workflow

### 1. Identify The Starting Point

Search the target recipe before editing:

```bash
rg -n "pack|packing|segment|position_ids|attention_mask|skip_mode|source_count" recipes/<recipe> mvp_engine
```

Then choose the mode:

- If the recipe already follows `basic_vlm`, read the relevant existing layer and make a local change.
- If the recipe has no packing, use `recipes/basic_vlm` as the implementation reference.
- If the user asks for model-specific behavior, inspect `model/packing/` and the model builder before changing dataset code.
- If the user asks for data/schema behavior, inspect `dataset/packing.py`, `dataset/collator.py`, and `dataset/dataset.py` first.

### 2. Identify Dataset Backend

Before copying any pipeline shape, identify whether the target recipe uses `mvp_dataset`.

Search for dataset backend markers:

```bash
rg -n "mvp_dataset|Dataset\\.assemble|RuntimeContext|TorchLoader|IterableDataset|torch\\.utils\\.data|datasets\\.Dataset|DataPipe" recipes/<recipe> mvp_engine
```

Reference in `basic_vlm`:

- `recipes/basic_vlm/dataset/dataset.py` uses `mvp_dataset.Dataset`, `Dataset.assemble`, `RuntimeContext`, `resolve_ref`, and deferred finalization.
- `recipes/basic_vlm/engine/basic_vlm_engine.py` wraps the dataset with `mvp_dataset.TorchLoader`.
- `recipes/basic_vlm/dataset/packing.py` relies on the `Assembler.push/finish` lifecycle and `RuntimeContext.sample_shuffle_seed`.
- `recipes/basic_vlm/dataset/skip.py` relies on `RuntimeContext.slot` for post-pack fast-resume markers.

If the target recipe does not use `mvp_dataset`, do not assume `Assembler`, `RuntimeContext`, `resolve_ref`, worker slots, or `TorchLoader` exist. Re-decide packing lifecycle, worker-local seeding, resume boundaries, materialization points, and step inference for that backend.

### 3. Config Layer

For existing `basic_vlm`-style recipes, preserve the current config shape unless the user needs a new behavior.

Reference in `basic_vlm`:

- `recipes/basic_vlm/configs/schema.py` defines the packing fields and validators.
- `recipes/basic_vlm/configs/stage*.yaml` shows which training stages enable packing and which knobs are active.

Typical knobs are:

```python
packing: bool = False
packing_selection_strategy: Literal["random", "best_fit"] = "best_fit"
packing_open_pack_limit: int = Field(8, ge=1)
packing_buffer_size: int = Field(64, ge=0)
```

Guidelines:

- Keep packing disabled by default unless the recipe's active configs intentionally enable it.
- Add only knobs that are used by active code or required for a reproducible experiment.
- Update only YAML configs that should change behavior.

### 4. Dataset Pipeline Layer

For `mvp_dataset` pipelines, prefer this delayed-materialization shape:

Reference in `basic_vlm`:

- `recipes/basic_vlm/dataset/dataset.py` owns the transform order, skip modes, and pack/finalize placement.
- `recipes/basic_vlm/engine/basic_vlm_engine.py` calls the dataset builder differently for normal training, step inference, and fast resume.

```text
raw row -> guard -> tokenize -> guard -> pack/group -> post-pack skip marker
        -> resolve refs/materialize payloads -> guard -> finalize pack -> collate
```

Guidelines:

- Pack after tokenization, when `input_ids` length and labels are known.
- If multimodal payload loading is expensive, group before loading and finalize after loading.
- If all payloads are already cheap tensors, finalization can happen immediately.
- Keep skip markers at the same output boundary consumed by training.
- Preserve the recipe's existing policy for invalid, empty, overlength, or unsupervised samples.

For non-`mvp_dataset` pipelines, explicitly choose the placement instead of copying the `basic_vlm` chain:

- Put the packer in an iterable dataset wrapper, dataset `__iter__`, a precompute/materialization step, or a pre-collator transform; avoid putting packing inside the collator unless the recipe deliberately treats one collated batch as the packing boundary.
- Define flush, finish, and `drop_last` behavior yourself; there may be no `Assembler.finish(drop_last=...)` callback.
- Derive deterministic worker-local seeds from `torch.utils.data.get_worker_info()`, data-parallel rank, epoch, and the recipe seed.
- Redesign fast resume; do not assume a stable `worker_slot` marker pass exists.
- Re-decide when image/audio/video references are resolved or tensors are materialized; do not assume `resolve_ref`.
- Ensure sharding, shuffle, and `drop_last` happen at the intended raw-sample or packed-sample boundary.

### 5. Packer And Finalizer Layer

The packer groups samples up to `max_seq_len`. The finalizer turns a group into one model-facing sample.

Reference in `basic_vlm`:

- `recipes/basic_vlm/dataset/packing.py` owns `PackedSampleAssembler`, `finalize_packed_sample_group`, packing metadata, and source-order payload merging.
- `skills/data/vlm-packing/references/packed-sample-assembler.md` gives generic assembler background, but the current production behavior is in `recipes/basic_vlm/dataset/packing.py`.

Responsibilities:

- Treat already-overlength samples according to the recipe policy, usually as standalone outputs.
- Concatenate token fields such as `input_ids`, `labels`, and `attention_mask`.
- Merge payload fields such as image tensors, image grids, references, or logging metadata in source order.
- Build compact boundary metadata used later by model preparation.
- Avoid carrying raw text, large unused payloads, or stale metadata into model batches.
- Use actual dataset runtime context fields for worker-local seeds; for non-`mvp_dataset`, inspect the active PyTorch/HF/custom dataset backend instead of guessing names.

When modifying a `basic_vlm`-derived recipe, start from `recipes/basic_vlm/dataset/packing.py`; use `references/packed-sample-assembler.md` only for generic background.

### 6. Collator Layer

The collator is responsible for producing a safe batch shape.

Reference in `basic_vlm`:

- `recipes/basic_vlm/dataset/collator.py` owns token padding, label padding, packed metadata padding, multimodal batching, and mixed packed/unpacked rejection.
- `recipes/basic_vlm/dataset/types.py` documents the model-facing batch fields.

Guidelines:

- Pad token tensors with the existing pad token and ignored-label values.
- Pad boundary metadata with an inactive value that cannot be confused with a real segment.
- Reject mixed packed/unpacked batches unless the recipe explicitly supports them.
- Keep text-only and multimodal samples compatible.
- If dummy payloads are appended for model compatibility, ensure their labels are ignored and their packed metadata remains isolated.

Example pattern:

```python
if any("boundary_ids" in sample for sample in batch):
    if not all("boundary_ids" in sample for sample in batch):
        raise ValueError("Packed and unpacked samples cannot be mixed in one batch.")
```

If a non-`mvp_dataset` recipe packs inside the collator, explicitly document that the collator is now both packer and batcher, then recheck token accounting, step inference, and resume because the packed-sample boundary is no longer independent of the batch boundary.

### 7. Model Preparation Layer

This layer is model-specific. Do not replace it with a text-only assumption for a VLM.

Reference in `basic_vlm`:

- `recipes/basic_vlm/model/packing/prepare.py` prepares packed model inputs before forward.
- `recipes/basic_vlm/model/packing/qwen3_vl.py` builds Qwen3-VL packed position ids.
- `recipes/basic_vlm/model/packing/fa2_patch.py` patches FlashAttention-2 paths for packed attention.
- `recipes/basic_vlm/model/qwen3_vl.py` owns the model builder and Qwen3-VL compatibility/loss patches.
- `recipes/basic_vlm/engine/basic_vlm_engine.py` calls packed-input preparation in `train_pre_step`.

Guidelines:

- Convert packed metadata into the exact representation needed by the model: block causal masks, packed position ids, sequence lengths, or backend-specific metadata.
- Preserve the model's multimodal position and RoPE rules inside each source segment.
- Ensure image/video grid metadata is consumed in the same order as packed placeholders.
- Patch every attention path that may transform or discard packed metadata.
- Preserve loss masking so no token is supervised across source-sample boundaries.
- Validate both padded batches and batches with mixed text-only/multimodal sources.

For `basic_vlm`, inspect `recipes/basic_vlm/model/packing/` before editing dataset packing for model-facing issues.

### 8. Training Accounting

Packing changes the unit consumed by training. Keep all training accounting on that same unit.

Reference in `basic_vlm`:

- `recipes/basic_vlm/utils/misc.py` owns packed total-step inference and batch-size/gradient-accumulation resolution.
- `recipes/basic_vlm/dataset/skip.py` owns post-pack skip marker helpers.
- `recipes/basic_vlm/engine/basic_vlm_engine.py` owns fast-resume marker passes, token accounting, loss normalization, and logging.

Check:

- total-step inference counts packed outputs, not raw rows;
- fast resume skips post-pack micro-batches consistently;
- token counts, loss normalization, gradient accumulation, throughput logging, and MFU logging still use the intended token totals;
- lightweight marker passes do not repeat expensive payload IO;
- late materialization failures cannot silently change the number of consumed packed outputs.
- for non-`mvp_dataset` backends, document whether resume and step inference are based on raw row offsets, shard offsets, global packed-sample indices, or unsupported.

### 9. Common Change Patterns

- **Change packing efficiency:** edit `dataset/packing.py` and relevant config knobs.
- **Change padding or batch validation:** edit `dataset/collator.py`.
- **Change packed attention or positions:** edit `model/packing/` and verify the engine still calls preparation before forward.
- **Change resume behavior:** inspect `dataset/skip.py`, `dataset/dataset.py`, and `engine/basic_vlm_engine.py` together.
- **Adapt to a new model:** start from `model/packing/`, then adjust finalizer metadata only if the model needs different inputs.
- **Adapt to a new data schema:** start from preprocessors and `dataset/packing.py`, then verify collator and model preparation still receive aligned payloads.
- **Adapt to a new dataset backend:** start from dataset construction and loader code, then decide packing lifecycle, seed source, materialization point, resume boundary, and packed-output counting before touching model code.

## Validation

For documentation-only or small local changes, run at least syntax/import checks that do not require GPUs.

For behavior changes, validate the affected layer:

- Config accepts intended YAMLs and rejects invalid knobs.
- Packer never exceeds the configured max length except for explicitly allowed standalone overlength samples.
- Labels, masks, payloads, and boundary metadata stay aligned with `input_ids`.
- Collator pads metadata correctly and handles text-only/multimodal batches.
- Model preparation blocks cross-source attention and preserves VLM-specific position rules.
- Step inference and fast resume count packed outputs consistently.
- For non-`mvp_dataset` backends, lifecycle, seeding, materialization, resume, and accounting decisions are documented in the change summary.

Do not add tests unless the user asks or the repository rules require them.

## Output

When reporting a change, state:

- whether you used `basic_vlm` as existing implementation or ported packing into another recipe;
- which layer was changed;
- the packing point, finalization point, and boundary metadata involved;
- what validation ran and what remains unverified.

## Read On Demand

- For concrete current behavior, read the relevant files under `recipes/basic_vlm`.
