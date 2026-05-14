---
name: vlm-packing
description: Add or adapt recipe-local sample or sequence packing for text and multimodal training recipes. Use when a recipe needs to combine multiple processed samples into longer model inputs while preserving loss boundaries, attention isolation, optional multimodal payloads, and recipe-specific model input preparation without moving the implementation into mvp_engine core.
---

# VLM Packing

## Goal

- Add recipe-local sample or sequence packing that reduces padding waste while preserving training semantics.
- Keep the implementation shaped by the target recipe's data schema, model interface, and attention backend.
- Do not add a shared runtime API or modify `mvp_engine/` unless the user explicitly asks.

Packing combines multiple already-processed source samples into one longer model input, usually up to the recipe's
maximum sequence length. The design starts from tokenized samples whose lengths and loss masks are known, groups them
with a simple recipe-local policy, concatenates token-aligned fields in source order, carries compact boundary metadata
such as segment ids, and merges optional multimodal payloads in the same order as their placeholders. The collator pads
both normal tensors and packing metadata, and the model-input preparation layer turns the boundary metadata into the
model-specific attention, position-id, or backend-specific representation needed to prevent cross-sample leakage.

Overall, design packing as a five-layer path: config exposes only the knobs the recipe needs; the dataset pipeline
chooses the packing and finalization points; the packer/finalizer owns grouping, concatenation, and boundary metadata;
the collator batches packed samples safely; and model-input preparation adapts packed metadata to the target model and
attention backend. Training accounting, resume, and step inference must count the same packed output unit consumed by
training.

## Required Inputs

- The target recipe path under `recipes/<recipe>/`.
- The recipe config schema and launch YAMLs that control data loading.
- The dataset pipeline, especially preprocessing order, IO/materialization steps, and filtering guards.
- The sample schema after tokenization, including token ids, labels, masks, optional multimodal tensors or references, and metadata.
- The collator and model/engine entrypoint that convert batches into model-ready inputs.
- The model's packed-input requirements, such as attention masks, position ids, loss masks, sequence lengths, or segment metadata.

## Workflow

### 1. Gather Context And Existing Support

- Read the recipe dataset builder, preprocessors, collator, config schema, engine, and model input preparation code.
- Preserve the recipe's existing module layout and naming style. If a recipe already imports a package such as
  `model/packing/`, extend that package instead of replacing it with a same-named module file.
- Search for existing packing or packed-attention support before adding new code:

```bash
rg -n "pack|packing|segment|position_ids|attention_mask|cu_seqlens|source_sample" recipes/<recipe> mvp_engine
```

If the recipe already has packing, first map it across the five layers below before changing code.

### 2. Layer 1: Recipe Config

- Add a disabled-by-default packing switch unless the target recipe already expects packing.
- Expose only knobs the recipe needs, typically maximum packed length, selection strategy, open-pack limit, and buffer size.
- Validate config values in the recipe schema.
- Update only YAML configs that should intentionally enable packing.
- Remove or avoid knobs that no active config uses unless they are needed for reproducible experiments.

Example config shape:

```python
packing: bool = False
packing_selection_strategy: Literal["best_fit", "random"] = "best_fit"
packing_open_pack_limit: int = Field(8, ge=1)
packing_buffer_size: int = Field(64, ge=0)
```

### 3. Layer 2: Dataset Pipeline

- Prefer packing after tokenization, when sample lengths are known.
- If samples contain expensive or external payloads, group before heavy materialization and finalize after payloads are loaded.
- If every model-facing field is already materialized cheaply, finalize packs immediately after grouping.
- Keep the deferred-pack representation explicit and easy for guards, materializers, and skip/resume helpers to recognize.
- Preserve the recipe's invalid, empty, overlength, or unsupervised sample policy before deciding whether packing should drop, keep, truncate, or emit samples standalone.
- If packing changes the dataset output unit, place resume skip markers at the same boundary that training consumes.

For delayed-materialization multimodal pipelines, a common shape is:

```text
raw row -> guard -> tokenize -> guard -> pack/group -> optional post-pack skip
        -> resolve refs/materialize payloads -> guard -> finalize pack -> collate
```

### 4. Layer 3: Packer And Finalizer

- Add a small recipe-local assembler or dataset transform that groups processed samples up to the recipe's maximum length.
- Keep the selection policy simple. Best-fit and random placement are usually enough; avoid adding a dependency for bin packing.
- Treat samples at or above the maximum length as standalone outputs unless the recipe already truncates or drops them earlier.
- List the token-length field, usually `input_ids`.
- Concatenate token-aligned fields in source order, commonly `input_ids`, `labels`, `attention_mask`, token masks, loss masks, and per-token metadata.
- Merge non-token payloads in the same order as their placeholders, such as image tensors, grids, audio features, references, source ids, or logging metadata.
- Build boundary metadata needed by model preparation, such as segment ids, sample offsets, sequence lengths, source counts, or block ids.
- If finalization is delayed, emit groups in the same shape expected by later dataset transforms and provide one finalization step after those transforms.
- Use the dataset framework's actual runtime context fields for worker-local seeding. Inspect them in the current repo
  instead of guessing names such as `rank` or `worker_id`.
- Preserve compact source-count or sample-origin metadata only when it is used for metrics, logging, resume, or debugging.
- Do not carry raw text, large unused payloads, or stale per-sample metadata into model batches.

Minimal schema notes:

```python
TOKEN_FIELDS = ("input_ids", "labels", "attention_mask")
OPTIONAL_PAYLOAD_FIELDS = ("<recipe_media_tensor>", "<recipe_media_metadata>")
PACK_BOUNDARY_FIELD = "<recipe_boundary_field>"


def sample_length(sample: dict) -> int:
    return int(sample["input_ids"].shape[0])
```

For a detailed generic `PackedSampleAssembler` implementation, read
`references/packed-sample-assembler.md`. Use it as a starting point, then adapt naming, buffering, seeding, and resume
semantics to the target recipe.

Example finalizer:

```python
def finalize_packed_sample_group(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("Cannot finalize an empty packed sample group.")

    packed = {
        "input_ids": torch.cat([sample["input_ids"] for sample in samples], dim=0),
        "labels": torch.cat([sample["labels"] for sample in samples], dim=0),
        "attention_mask": torch.cat([sample["attention_mask"] for sample in samples], dim=0),
        "boundary_ids": torch.cat(  # Rename to the recipe's boundary field when needed.
            [
                torch.full_like(sample["input_ids"], fill_value=index + 1, dtype=torch.long)
                for index, sample in enumerate(samples)
            ],
            dim=0,
        ),
        "source_count": len(samples),
    }

    # Repeat for each optional payload field that must be merged in source order.
    payloads = [sample["<recipe_media_tensor>"] for sample in samples if sample.get("<recipe_media_tensor>") is not None]
    packed["<recipe_media_tensor>"] = torch.cat(payloads, dim=0) if payloads else None
    return packed
```

### 5. Layer 4: Collation

- Pad packed token tensors with the existing pad token and ignored-label values.
- Pad boundary metadata with an inactive value that cannot be confused with a real source segment.
- Reject batches that mix packed and unpacked samples unless the recipe explicitly supports that.
- Keep optional multimodal collation compatible with text-only and multimodal batches.
- Preserve existing dummy-input or fallback behavior only if the recipe already needs it for model compatibility.
- If dummy payloads are appended for model compatibility, keep loss masked and give any packed boundary metadata a valid isolated segment.

Example collator addition:

```python
if any("boundary_ids" in sample for sample in batch):
    if not all("boundary_ids" in sample for sample in batch):
        raise ValueError("Packed and unpacked samples cannot be mixed in one batch.")
    model_inputs["boundary_ids"] = pad_sequence(
        [sample["boundary_ids"] for sample in batch],
        batch_first=True,
        padding_value=0,
    )
```

### 6. Layer 5: Model Input Preparation

- Convert packing metadata into the exact model-facing representation required by the target model.
- Keep attention-backend-specific logic recipe-local. For example, a model may need a block causal mask, sequence-length metadata, packed position ids, or a backend patch.
- For multimodal models, do not replace model-specific position-id or RoPE logic with a text-only reset unless the model
  actually uses text-only positions. Mirror the model's image/video grid handling and verify that multimodal metadata is
  consumed in the same order as packed placeholders.
- For backend patches, patch every path that can transform or discard the packed boundary metadata. Some models build
  causal masks in model-specific helpers before calling shared attention utilities, so patching only the shared unpad
  helper may be insufficient.
- Preserve loss masking so tokens from different source samples do not create cross-sample supervised targets.
- Validate that packed inputs still work when the batch contains padding and when some sources have no multimodal payload.

Example block causal mask helper:

```python
def build_packed_block_causal_mask(boundary_ids: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if boundary_ids.ndim != 2:
        raise ValueError(f"Expected 2D boundary_ids, got shape {tuple(boundary_ids.shape)}.")

    batch_size, sequence_length = boundary_ids.shape
    positions = torch.arange(sequence_length, device=boundary_ids.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    valid = boundary_ids.ne(0)
    same_source = boundary_ids.unsqueeze(-1) == boundary_ids.unsqueeze(-2)
    allowed = valid.unsqueeze(-1) & valid.unsqueeze(-2) & same_source & causal.unsqueeze(0)

    mask = torch.full(
        (batch_size, 1, sequence_length, sequence_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=boundary_ids.device,
    )
    return mask.masked_fill(allowed.unsqueeze(1), 0)
```

### 7. Cross-Layer Training Accounting

- Confirm token counting, loss normalization, gradient accumulation, resume skipping, and throughput logging still count the intended unit.
- If the recipe resumes by dataloader position, decide whether resume boundaries are pre-pack or post-pack and implement one consistent policy.
- If the recipe infers total steps from the dataset, count packed outputs rather than raw source rows.
- If adding packing changes the dataloader output unit, update any fast-resume or skip logic to skip the same unit that
  training consumes. For post-pack resume, use a lightweight marker pass that does not repeat expensive payload IO.
- If step inference or fast resume skips payload materialization, verify late materialization failures cannot change the consumed output count, or document the residual risk.

## Validation

- Config: packing is disabled by default unless the target recipe intentionally enables it, and active YAMLs use only supported knobs.
- Dataset pipeline: packing point, finalization point, guards, materialization, and resume skip boundaries consume the same output unit.
- Packer/finalizer: packed outputs never exceed the configured maximum length except for already-overlength standalone samples.
- Packer/finalizer: labels, masks, payloads, and boundary metadata stay aligned with `input_ids`.
- Collator: packed metadata pads with an inactive value, mixed packed/unpacked batches are handled deliberately, and dummy payload fallback preserves loss and segment isolation.
- Model preparation: cross-source attention is blocked or represented according to model requirements.
- Model preparation: model-specific position ids match the unpacked model's rules inside each source segment, including image/video grid positions when present.
- Model preparation: backend-specific patches preserve packed boundary metadata through model-specific mask creation and shared attention utility paths.
- Training accounting: resume and total-step inference count packed outputs when packing is enabled.
- Run the smallest available syntax or smoke validation for the changed recipe files.
- Add recipe-local tests only when the user or repository rules allow tests.

## Output

- State which recipe files were updated and whether packing is enabled in any YAML config.
- State the packing point, finalization point, and boundary metadata used.
- State how the five layers are wired: config, dataset pipeline, packer/finalizer, collator, and model input preparation.
- State what validation ran and what remains unverified.

## Read On Demand

- For detailed generic assembler code, read `references/packed-sample-assembler.md`.
- If a recipe has local packing notes or reference implementations, read only the relevant file for that recipe.
