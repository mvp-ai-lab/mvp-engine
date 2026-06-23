# MLLM Schema And Tokenization

Use this reference when raw rows, prompts, placeholders, roles, or label policy
need source- or model-family-specific normalization.

## Contract

`MLLMSchemaHandler.normalize(row)` returns:

```python
(
    list[MLLMSegment],
    list[MLLMMediaSlot],
    dict[str, Any],
)
```

`MLLMSegment` is the ordered stream unit:

```python
MLLMSegment(type="text", loss=False, value="prompt text")
MLLMSegment(type="text", loss=True, value="target text")
MLLMSegment(type="image", loss=False, value="image:0")
MLLMSegment(type="video", loss=False, value="video:0")
```

Segment fields:

- `type`: `"text"` for literal text, or a media type registered in
  `MLLMMediaHandler`.
- `loss`: whether tokens produced from this segment become supervised labels.
- `value`: text content for text segments; media id for media segments before
  media rendering.

`MLLMMediaSlot` binds a media id to a raw source field:

```python
MLLMMediaSlot(
    media_id="image:0",
    media_type="image",
    field="images",
    index=0,
    metadata={"size": [height, width]},
)
```

Slot fields:

- `media_id`: stable id referenced by media segments.
- `media_type`: media handler key, such as `"image"` or `"video"`.
- `field`: raw sample field read by `MLLMMediaHandler.load(...)`.
- `index`: optional index inside a sequence field.
- `metadata`: lightweight information needed before media IO, such as size,
  duration, frame count, or token-count hints.

## Normalization Steps

When implementing a schema handler:

1. Copy or read the raw row without mutating external dataset state.
2. Validate cheap schema errors: role shape, text type, media field shape, media
   count, and required metadata.
3. Normalize role aliases and message field aliases.
4. Bind raw media values into ordered `MLLMMediaSlot` objects.
5. Split raw text on media placeholders and insert media segments with matching
   `media_id`.
6. Apply model chat templates or source/target rendering.
7. Emit final `MLLMSegment` objects with explicit `loss` flags.
8. Return small metadata only when another component uses it.

Tokenization, media IO, tensor construction, and distributed state handling live
outside schema handlers.

## Label Policy

Schema is the source of truth for supervised regions. Media and tokenization
code follow segment loss flags.

Common policies:

- Chat SFT: user/system/tool/source template text `loss=False`; assistant target
  text `loss=True`.
- Captioning: optional prompt text `loss=False`; caption text `loss=True`.
- Interleaved full-LM: trainable text spans `loss=True`; media placeholder
  segments `loss=False`.
- Filtering or instruction variants: encode the decision in emitted segments so
  tokenizer-side masking stays mechanical.

Example caption row:

```python
segments = [
    MLLMSegment(type="text", loss=False, value="Describe the image.\n"),
    MLLMSegment(type="image", loss=False, value="image:0"),
    MLLMSegment(type="text", loss=True, value="A red car parked beside a tree."),
]
media_slots = [
    MLLMMediaSlot("image:0", "image", field="images", index=0, metadata={"size": [720, 1280]}),
]
```

Example interleaved row:

```python
segments = [
    MLLMSegment(type="text", loss=True, value="A recipe begins with "),
    MLLMSegment(type="image", loss=False, value="image:0"),
    MLLMSegment(type="text", loss=True, value=" and then continues with the next step."),
]
```

## Tokenization

`MLLMTokenizationHandler.tokenize(rendered_segments)`:

- expects every segment value to be rendered text;
- tokenizes each segment independently without special tokens;
- copies token ids into labels only when `segment.loss` is true;
- fills ignored labels with `ignore_index`;
- rejects truncation through a non-text media segment.

Implement a custom tokenization handler when a model needs different special
token handling, media truncation rules, or label conversion.

## Qwen Reference

`data_kit.QwenVLChatSchemaHandler` supports conversation-style rows with:

- `messages` or `conversations`;
- role/content pairs using either `role`/`content` or `from`/`value`;
- ordered image placeholders such as `<image>`, Qwen image token, or wrapped Qwen
  vision token;
- image refs from either `images` plus `img_size`/`image_size`, or explicit
  `media` entries.

It applies the processor chat template, splits source and assistant target text,
handles Qwen thinking-mode normalization, and emits image segments with
`loss=False`.

## Implementation Checklist

- Placeholder count matches media slot count.
- Every media segment value matches one `MLLMMediaSlot.media_id`.
- Media slots use raw fields that remain available on `MLLMSample`.
- Size or token-count metadata is present when media rendering needs it.
- Loss flags match the training objective before tokenization.
- Empty or unsupported rows fail early with clear `ValueError`.
- Handler output can be consumed by `MLLMMediaHandler.render(...)` and
  `MLLMTokenizationHandler.tokenize(...)`.
