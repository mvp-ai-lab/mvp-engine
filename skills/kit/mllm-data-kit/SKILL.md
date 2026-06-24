---
name: mllm-data-kit
description: Use and extend the MVP-Engine MLLM data kit with explicit specs,
  handlers, and sample objects. Use for recipe data wiring, MLLMDataSpec
  construction, source resample/resolve_refs policy, schema/media/tokenization
  handlers, packing, guards, collation, dataloader setup, Qwen VL data support,
  and multimodal data extensions.
---

# MLLM Data Kit

## Goal

Use `MLLMDataKit` as the standard data setup API for MLLM recipes. The kit
builds a complete mvp-dataset pipeline from explicit specs:

- `build_processor(...)` loads and normalizes the processor;
- `build_distribution_spec(...)` derives data-parallel mesh placement;
- `build_dataset(...)` builds source, guards, sample wrapping, packing,
  reference resolution, and model-input conversion;
- `build_collator(...)` pads packed samples and delegates media collation;
- `build_dataloader(...)` wraps the dataset in a batched `TorchLoader`;
- `to_device(...)` moves tensor batch fields to the training device.

The design has three layers:

- specs declare pipeline choices;
- handlers define source-, model-, and modality-specific behavior;
- `MLLMSample` and `MLLMPack` own sample state and packed model-input assembly.

## Required Inputs

Identify these before editing:

- target recipe engine and `prepare_dataloader()` path;
- dataset path, source backend, ref columns, seed, and DP sharding behavior;
- train and estimation source behavior: `resample` and `resolve_refs`;
- raw row schema, role aliases, media placeholder convention, and label policy;
- processor path and model-facing media tensor fields;
- max sequence length, packing strategy, open-pack limit, and buffer size;
- batch size, worker count, pin-memory policy, and text-only batch requirements;
- recipe-local packed model-input preparation path;
- recipe structure/smoke validation commands.

Ask the user only when raw schema semantics, media tensor contract, or dataset
backend lifecycle cannot be derived locally.

## Workflow

### 1. Declare The Components

Build specs directly in the recipe engine:

```python
from mvp_engine.kit import (
    MLLMDataKit,
    MLLMDataSpec,
    MLLMLoaderSpec,
    MLLMPackingSpec,
    MLLMSampleSpec,
    MLLMSourceSpec,
    QwenVLChatSchemaHandler,
    QwenVLMediaHandler,
    QwenVLTokenizationHandler,
)

self.data_kit = MLLMDataKit()
processor = self.data_kit.build_processor(...)

sample_spec = MLLMSampleSpec(
    schema_handler=QwenVLChatSchemaHandler(processor=processor, thinking_mode=config.data.thinking_mode),
    media_handler=QwenVLMediaHandler(processor=processor),
    tokenization_handler=QwenVLTokenizationHandler(
        processor=processor,
        max_seq_len=int(config.data.max_seq_len),
    ),
)
packing_spec = MLLMPackingSpec(
    max_seq_len=int(config.data.max_seq_len),
    algorithm="multi_pack",
    selection_strategy=config.data.packing_selection_strategy,
    open_pack_limit=int(config.data.packing_open_pack_limit),
    buffer_size=int(config.data.packing_buffer_size),
    block_causal=True,
)
loader_spec = MLLMLoaderSpec(
    batch_size=int(config.data.batch_size),
    num_workers=int(config.data.num_workers),
)
distribution = self.data_kit.build_distribution_spec(parallel_mesh=self.parallel_mesh)
```

### 2. Build Train And Estimation Specs if Needed

Train data usually resolves refs and resamples:

```python
train_spec = MLLMDataSpec(
    source=MLLMSourceSpec(
        dataset_path=config.data.train_path,
        dataset_source="lance",
        ref_columns=tuple(config.data.ref_columns),
        seed=int(config.seed),
        resample=True,
        resolve_refs=True,
    ),
    sample=sample_spec,
    packing=packing_spec,
    loader=loader_spec,
    distribution=distribution,
)
```

Step estimation usually consumes a finite packed stream without resolving media:

```python
estimate_spec = MLLMDataSpec(
    source=MLLMSourceSpec(
        dataset_path=config.data.train_path,
        dataset_source="lance",
        ref_columns=tuple(config.data.ref_columns),
        seed=int(config.seed),
        resample=False,
        resolve_refs=False,
    ),
    sample=sample_spec,
    packing=packing_spec,
    loader=loader_spec,
    distribution=distribution,
)
```

Pass `data_kit.build_dataset(estimate_spec)` to
`MLLMStepEstimationKit.estimate_total_steps(...)` when `loop.total_steps == -1`.

### 3. Build The Runtime Pipeline

Use the kit for dataset and loader construction:

```python
dataset = self.data_kit.build_dataset(train_spec)
dataloader = self.data_kit.build_dataloader(dataset, train_spec, device=self.device)
```

`build_dataset(spec)` creates this flow:

```text
Dataset.from_source
-> MLLMRawRowGuard
-> MLLMSample.from_row(sample_spec)
-> MLLMSampleGuard
-> MLLMPackingAssembler
-> resolve_ref, if source.resolve_refs and source.ref_columns
-> MLLMPack.to_model_inputs()
-> MLLMModelInputGuard
```

The collator then pads token fields, pads `pack_segment_ids` with `0`, creates
token counters, and delegates media fields to `MLLMMediaHandler.collate(...)`.

### 4. Extend At The Smallest Boundary

- New raw schema, prompt rendering, media slot binding, or label policy:
  implement `MLLMSchemaHandler`.
- New media type or model media contract: implement `MLLMMediaTypeHandler` and
  register it in `MLLMMediaHandler`.
- New tokenization or truncation behavior: implement `MLLMTokenizationHandler`.
- New packing algorithm: pass a custom assembler with
  `MLLMPackingSpec(assembler_cls=...)`.
- New text-only batch requirement: attach `MLLMTextOnlyBatchGuard` or a
  recipe-local loader map after collation.
- New dataset backend or stage order: extend `MLLMDataKit`.

### 5. Keep Model-Specific Packed Preparation In The Recipe

Generic DataKit packing produces `pack_segment_ids`, `source_sample_num`, token
fields, and merged media tensors. Recipe/model code should prepare backend
attention masks, position ids, FlashAttention metadata, multimodal RoPE rules,
and other model-forward details.

### 6. Know The Component Responsibilities

#### Specs

- `MLLMSourceSpec`: dataset path, source type, ref columns, seed, `resample`,
  and `resolve_refs`.
- `MLLMSampleSpec`: schema, media, and tokenization handlers.
- `MLLMPackingSpec`: max length, algorithm name, selection strategy,
  open-pack limit, buffer size, block-causal flag, and custom assembler.
- `MLLMLoaderSpec`: batch size, worker count, pin-memory behavior, worker
  persistence, multiprocessing context, and `drop_last`.
- `MLLMDistributionSpec`: device mesh and data-parallel mesh dimensions.
- `MLLMDataSpec`: full source/sample/packing/loader/distribution declaration.

#### Handlers

- `MLLMSchemaHandler.normalize(row)` returns `MLLMSegment` objects,
  `MLLMMediaSlot` objects, and metadata. It owns source format support, media
  slot binding, segment order, and label policy.
- `MLLMMediaTypeHandler` owns one modality's placeholder rendering, media IO,
  pack merge, and batch collation.
- `MLLMMediaHandler` dispatches registered media type handlers.
- `MLLMTokenizationHandler` tokenizes rendered segments and converts
  `segment.loss` into labels.

#### Runtime Objects

- `MLLMSample` wraps one raw row and lazily materializes segments, token fields,
  media slots, loaded media tensors, and model inputs.
- `MLLMPackingAssembler` groups tokenized samples with bounded buffering.
- `MLLMPack` loads media after refs are resolved and emits one packed
  model-input dict.
- `MLLMBatchCollator` pads packed samples and computes token/source counters.
- `MLLMRawRowGuard`, `MLLMSampleGuard`, and `MLLMModelInputGuard` filter invalid
  data at pipeline boundaries.
- `MLLMTextOnlyBatchGuard` adds dummy media fields for model backends that need
  a non-empty media path.

#### Qwen Components

- `QwenVLChatSchemaHandler`: conversation rows, Qwen chat template rendering,
  thinking-mode handling, and ordered image slots.
- `QwenVLMediaHandler`: Qwen media registry with image support.
- `QwenImageHandler`: Qwen image placeholder expansion, smart resize, image
  loading, pack merge, batch collation, and dummy image inputs.
- `QwenVLTokenizationHandler`: Qwen-facing alias for the standard tokenization
  handler.

## Validation

### Soft Validation

- recipe constructs explicit source, sample, packing, loader, and distribution specs;
- train and estimation specs set `resample` and `resolve_refs` explicitly;
- schema handlers emit ordered `MLLMSegment(type, loss, value)` and `MLLMMediaSlot`;
- media handlers own placeholder rendering, media IO, pack merge, and batch collation;
- packed model inputs contain `input_ids`, `attention_mask`, `labels`,
  `pack_segment_ids`, `source_sample_num`, and expected media tensors;
- model-specific attention masks, position ids, and FlashAttention metadata stay
  in recipe/model preparation code;
- invalid rows, empty tokenization, unreadable media, and text-only batches have
  explicit behavior.

### Hard Validation

Run the recipe's normal structure and smoke tests when the environment supports
them.

## Output

- State which specs and handlers are used.
- State custom schema, media, tokenization, packing, guard, or DataKit changes.
- State source schema, media lifecycle, packing knobs, and collator outputs.
- State model-specific packed input preparation.
- Report validation commands and remaining untested modality cases.

## Read On Demand

- `references/schema-transform.md`: schema, segment, tokenization, and label rules.
- `references/media.md`: media rendering, loading, merge, and collation rules.
- `references/packing.md`: standard packing and packed metadata rules.
