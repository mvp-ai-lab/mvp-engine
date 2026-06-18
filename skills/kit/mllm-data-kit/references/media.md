# MLLM Media Handling

Use this reference when adding or reviewing image, video, audio, or other media
support.

## Components

- `MLLMMediaTypeHandler`: behavior for one media type.
- `MLLMMediaHandler`: registry and dispatcher over media type handlers.
- `RenderedMedia`: placeholder text produced for one `MLLMMediaSlot`.
- `MLLMSample.load_media()`: loads model media fields after refs are resolved.
- `MLLMPack.to_model_inputs()`: merges loaded media across samples in one pack.
- `MLLMBatchCollator`: calls `media_handler.collate(batch)`.

## Lifecycle

```text
schema handler: raw row -> media slots and media segments
media handler render: media id -> placeholder text
tokenization handler: rendered segments -> tokens and labels
dataset resolve_ref: raw media refs become usable values, if enabled
media handler load: raw media values -> model media tensors
media handler merge_pack: source sample tensors -> packed sample tensors
media handler collate: packed sample tensors -> batch tensors
```

Place expensive IO in `MLLMMediaTypeHandler.load(...)`. Schema normalization and
placeholder rendering stay lightweight.

## Render Contract

`render(slot, *, processor, tokenizer) -> RenderedMedia` expands one media slot
into text that the tokenizer will see.

`RenderedMedia` fields:

- `media_id`: copied from the slot.
- `media_type`: handler media type.
- `text`: model-specific placeholder text, such as repeated image tokens.
- `metadata`: optional render metadata.

Rendering should use slot metadata and processor geometry when token length
depends on media shape. For Qwen image data, this is where smart-resized image
height and width become repeated image placeholder tokens.

## Load Contract

`load(slots, values, *, processor) -> dict[str, Any]` converts raw values into
model media fields.

Return semantics:

- `{}`: this media type contributes no tensor fields for this sample.
- `{"pixel_values": ..., "image_grid_thw": ...}` or equivalent: loaded model media.
- `empty_model_sample()`: the whole sample is unusable and should be dropped.

Use `{}` for valid text-only cases or optional unresolved media paths that should
not invalidate the sample. Use `empty_model_sample()` for corrupt or unreadable
media when training should skip the sample.

## Pack And Batch Contracts

`merge_pack(samples)` receives per-source-sample model-input dictionaries inside
one `MLLMPack`. Concatenate or merge media fields in source-sample order.

`collate(batch)` receives finalized packed dictionaries inside one batch.
Concatenate, pad, or stack media fields exactly as the model forward expects.

The token collator handles token padding and counters; media handlers only add
media-specific fields.

## Registering Media Types

Register handlers by media type:

```python
media_handler = MLLMMediaHandler(
    processor=processor,
    handlers={
        "image": MyImageHandler(),
        "video": MyVideoHandler(),
    },
)
```

Schema handlers should emit media segments and slots using those same keys:

```python
MLLMSegment(type="video", loss=False, value="video:0")
MLLMMediaSlot(media_id="video:0", media_type="video", field="videos", index=0)
```

## Ownership

- `MLLMDataKit` stays model-family agnostic.
- Media handlers own model-family tokens, image resizing, video sampling, and
  tensor field names.
- Schema emits `loss`; tokenizer applies it; media handlers focus on rendering
  and tensor preparation.
- Media loading reads the sample's current raw fields. After `resolve_ref`, those
  fields may already contain bytes or decoded backend values.
- Pack merge and batch collation preserve sample and placeholder order.
- Model-backend dummy-media behavior belongs in `MLLMTextOnlyBatchGuard` or a
  recipe-local loader map.

## Text-Only Batch Guard

Some VLM backends require at least one media tensor in every local batch. Attach
`MLLMTextOnlyBatchGuard` after collation when needed:

```python
dummy_inputs = QwenImageHandler().build_dummy_inputs(processor)
guard = MLLMTextOnlyBatchGuard(
    dummy_inputs=dummy_inputs,
    media_keys=("pixel_values", "image_grid_thw"),
    pad_token_id=processor.tokenizer.pad_token_id,
)
```

Use this as a recipe-local loader map or equivalent batch stage.

## Qwen Image Reference

`QwenVLMediaHandler` registers `QwenImageHandler` for `image` media.

`QwenImageHandler`:

- renders `<|vision_start|>` + repeated image token + `<|vision_end|>`;
- estimates token count from Qwen smart-resized height/width, patch size, and
  merge size;
- accepts paths, bytes, image records, or PIL images;
- emits `pixel_values` and `image_grid_thw`;
- concatenates those tensors at pack and batch boundaries;
- can build dummy image inputs for recipes that require text-only batch guards.

## Implementation Checklist

- Handler `media_type` matches emitted `MLLMSegment.type` and
  `MLLMMediaSlot.media_type`.
- Rendered media text tokenizes to the expected number of media tokens.
- `load(...)` handles every raw value shape produced by the source backend.
- Pack merge and batch collation keep media tensor order aligned with
  placeholders.
- Optional text-only behavior is explicit.
- Corrupt media either returns an empty-sample sentinel or has a documented
  recipe policy.
