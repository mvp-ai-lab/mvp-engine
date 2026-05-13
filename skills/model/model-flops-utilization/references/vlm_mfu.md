# VLM MFU

## Use When

Use this reference for vision-language models, multimodal Transformers,
image/video token insertion, dual-stream models, vision encoders, resamplers,
projectors, cross-attention, or frozen vision/text modules.

## Counting Rules

- Count all executed model components: vision encoder, projector/resampler,
  language model, cross-modal fusion, and output head.
- For the language model, use the actual sequence length after image or video
  tokens are inserted.
- For multi-image or video inputs, include every visual token that reaches the
  projector, resampler, cross-attention, or LLM.
- Count frozen modules as forward-only if they execute.
- Count trainable modules with the normal training multiplier when backward is
  computed through them.
- If a frozen module runs under `torch.no_grad()`, do not apply backward FLOPs
  to that module.

## Component Guidance

Vision encoder:

- For ViT-style encoders, count patch embedding, per-layer self-attention, MLP,
  and any classification or pooling head that actually runs.
- Use actual image resolution and patch layout when it can vary by batch.

Projector or resampler:

- Count linear projections, MLP projectors, Q-former/resampler attention, and
  pooling/fusion layers that transform visual features into language tokens.
- For cross-attention resamplers, use query length as resampler token count and
  key/value length as visual token count.

Language model:

- Count text tokens plus inserted visual tokens for self-attention and MLP.
- Count LM head over positions where logits are computed. If logits are only
  computed for a subset, use that subset length.

Dual-stream or fusion models:

- Count both streams that execute.
- Count stream initialization projections, per-layer fusion, cross-attention,
  concatenation, and final fusion layers.
- Do not double-count shared layers reused by both streams unless they execute
  separately for each stream.

## Common Mistakes

- Counting only the LLM and omitting the vision encoder or projector.
- Using text-only sequence length after visual tokens have been inserted.
- Applying full training FLOPs to frozen vision modules running under
  `torch.no_grad()`.
- Ignoring video frame count or multiple images when computing visual tokens.
- Counting preprocessing, image decode, or dataloader time as model FLOPs.
