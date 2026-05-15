# FLOPs Formulas

## Use When

Use this reference when implementing `calculate_model_flops(...)` for dense
Transformer, ViT, encoder-decoder, VLM, or custom model components.

## Counting Convention

- Count one multiply-add as 2 FLOPs.
- Count model FLOPs only. Do not include data loading, logging, checkpoint I/O,
  communication, dispatch, gather, scatter, or collective overhead.
- Default train FLOPs are `3 * forward_flops`: forward, backward through
  activations, and gradient computation.
- Add recomputed forward FLOPs for activation-checkpointed regions.
- If a module runs under `torch.no_grad()`, count forward FLOPs only.

## Core Formulas

- Linear `Din -> Dout` over `N` items: `2 * N * Din * Dout`.
- Q projection: `2 * B * T * D * D`.
- K/V projections: `2 * B * T * D * D` each for standard MHA.
- Attention QK: `2 * B * H * Tq * Tk * Dh`.
- Attention AV: `2 * B * H * Tq * Tk * Dh`.
- Dense causal self-attention matmuls: `4 * B * T * T * D`.
- Attention output projection: `2 * B * T * D * D`.
- Dense MLP `D -> F -> D`: `4 * B * T * D * F`.
- SwiGLU/GEGLU `D -> F`, `D -> F`, `F -> D`: `6 * B * T * D * F`.
- LM head `D -> vocab`: `2 * B * T * D * vocab`.
- ViT patch embedding: `2 * B * num_patches * (C * patch_h * patch_w) * D`.
- Classification head `D -> classes`: `2 * B * D * classes`.

## Architecture Notes

- For GQA/MQA, scale K/V projection FLOPs by `num_kv_heads / num_heads`.
  Do not scale Q projection, QK, AV, or output projection by that ratio.
- For encoder-decoder models, count encoder self-attention, decoder
  self-attention, decoder cross-attention, decoder MLP, and output head.
- For cross-attention, use source length as `Tk` and target length as `Tq`.
- For packed or variable-length batches, prefer actual token counts over
  `batch_size * max_seq_len`.
- For padded batches, count padded tokens only if the implementation actually
  computes them.

## Common Mistakes

- Do not use `6 * params * tokens` for custom architectures unless the user
  explicitly requests a rough dense-LM estimate.
- Do not count optimizer math, gradient clipping, communication, or data
  pipeline time as model FLOPs.
- Do not mix forward-only FLOPs with optimizer-step timing unless the metric is
  explicitly labeled as forward-only.
- Do not apply the training multiplier to frozen modules that do not run
  backward.
