# FLOPs Formulas

Use this reference when implementing `calculate_model_flops(...)` for dense
Transformers, ViTs, encoder-decoder models, VLMs, MoE models, sparse routing, or
custom model components.

## Counting Convention

- Count one multiply-add as 2 FLOPs.
- Standard MFU counts useful model FLOPs only. Exclude data loading, preprocessing, logging,
  checkpoint I/O, optimizer math, gradient clipping, dispatch, gather, scatter,
  all-to-all, and collectives.
- MFU is
  `model_flops_per_step / (step_time_seconds * sum_active_device_peak_flops)`.
- Use dense peak FLOPs for the active precision unless structured sparsity is
  actually enabled and the numerator uses the same sparsity convention.
- Default trainable matmul FLOPs are about `3 * forward_flops`: forward,
  activation/input-gradient matmul, and weight-gradient matmul.
- Do not include activation-checkpoint or rematerialization recompute in standard
  MFU. Include recompute only for HFU or actual executed model-math FLOPs, and
  label that metric separately.
- Frozen parameters remove weight-gradient FLOPs. If gradients still propagate
  through the frozen module, count activation/input-gradient backward FLOPs.
- Count frozen modules as forward-only only when no gradient propagates through
  them, such as `torch.no_grad()`, `torch.inference_mode()`, detached outputs,
  or no grad-requiring inputs.
- For packed or variable-length batches, use actual token counts for linear
  layers and actual attention-pair counts for attention.
- Count padded tokens only if the implementation actually computes them.

## Dense Building Blocks

- Linear `Din -> Dout` over `N` items: `2 * N * Din * Dout`.
- Q projection: `2 * B * T * D_in * (Hq * Dq)`.
- K projection: `2 * B * T * D_in * (Hkv * Dk)`.
- V projection: `2 * B * T * D_in * (Hkv * Dv)`.
- Attention QK: `2 * B * H * Tq * Tk * Dh`.
- Attention AV: `2 * B * H * Tq * Tk * Dh`.
- Full-square self-attention QK + AV, used by the common literature MFU
  convention: `4 * B * T * T * D`.
- Causal triangular QK + AV, when the kernel actually skips masked upper-triangle
  pairs: `2 * B * T * (T + 1) * D`.
- Attention output projection: `2 * B * T * (Hq * Dv) * D_out`.
- Dense MLP `D -> F -> D`: `4 * B * T * D * F`.
- SwiGLU/GEGLU `D -> F`, `D -> F`, `F -> D`: `6 * B * T * D * F`.
- LM head `D -> vocab`: `2 * B * T * D * vocab`.
- ViT patch embedding: `2 * B * num_patches * (C * patch_h * patch_w) * D`.
- Classification head `D -> classes`: `2 * B * D * classes`.

## Transformer Notes

- For GQA/MQA, scale only K/V projection FLOPs by
  `num_kv_heads / num_heads`. Do not scale Q projection, QK, AV, or output
  projection by that ratio.
- Standard MHA with `Hq * Dq = Hkv * Dk = Hkv * Dv = D` reduces Q/K/V/O
  projections to `2 * B * T * D * D` each.
- For encoder-decoder models, count encoder self-attention and encoder MLP,
  decoder self-attention, decoder cross-attention, decoder MLP, and output or
  task head where executed.
- For cross-attention, use source length as `Tk` and target length as `Tq`.
- For KV reuse, remove K/V projection FLOPs for layers that reuse external KV.
  Still count K/V projections for newly appended tokens whose KV are created in
  the current step. Keep Q projection, QK, AV, output projection, and fusion
  compute.

## Packed And Variable-Length Attention

- For linear layers, MLP, router, projector, and LM head, use actual token counts.
- For attention, count actual allowed attention pairs, not just total tokens.
- Full bidirectional per sample: use `sum_i L_i^2`.
- Causal triangular per sample: use `sum_i L_i * (L_i + 1) / 2`.
- Packed full-dense kernels may still materialize masked cross-sample pairs. In
  that case, count what the kernel computes, possibly `packed_T^2`.
- If packing masks block cross-sample attention and the kernel skips those
  pairs, use per-sample lengths inside the pack rather than the concatenated
  sequence length.

## VLM Rules

- Count every executed model component: vision encoder, projector/resampler,
  language model, cross-modal fusion, and output head.
- For the language model, use the actual sequence length after image or video
  tokens are inserted.
- For multi-image or video inputs, include every visual token that reaches the
  projector, resampler, cross-attention, or LLM.
- For ViT-style encoders, count patch embedding, per-layer self-attention, MLP,
  and any classification or pooling head that actually runs.
- Use actual image resolution, frame count, and patch layout when they vary by
  batch.
- For projectors and resamplers, count linear projections, MLP projectors,
  Q-former/resampler attention, and pooling or fusion layers.
- For cross-attention resamplers, use resampler query length as `Tq` and visual
  token length as `Tk`.
- Count LM head only over positions where logits are computed.
- For dual-stream models, count both streams and any initialization projection,
  fusion layer, cross-attention, concatenation, and final fusion that executes.
  Do not double-count shared layers unless they execute separately per stream.

## MoE And Sparse Routing Rules

- Count activated expert FLOPs, not total expert parameter FLOPs.
- Prefer router-produced per-expert token counts.
- For top-k routing, each token contributes to each selected expert path.
- Count router/gate linear FLOPs separately when material.
- Count shared dense experts for all tokens that pass through them.
- Count dropped tokens only for compute they actually execute.
- Count padded expert capacity only if the implementation performs expert
  compute on padded slots.
- For MoD or token routing, use selected token count `k` for routed attention,
  routed FFN, or sparse branch compute.
- Use full token count `T` for router scoring, token selection, fusion,
  residual projections, final output layers, and LM head unless those modules
  are also sparse.
- For shared dense paths plus sparse branches, count both paths for the tokens
  that execute them.
- If sparse branch token counts are unavailable, use configured selected token
  count or routing ratio and label the value as an estimate.

## Distributed MoE Rules

- For global MFU, count logical global routed tokens across data-parallel
  samples.
- Expert parallelism changes expert ownership, not the logical amount of model
  compute.
- Count activated experts logically, including experts owned by other ranks.
- Do not multiply routed tokens by `expert_parallel_size` unless EP also changes
  the number of input samples.
- Communication overhead lowers MFU through measured step time; it is not model
  FLOPs.
- Do not report rank-local routed FLOPs as global MFU unless the denominator is
  also rank-local and the metric is labeled separately.

## Common Mistakes

- Using `6 * params * tokens` for custom architectures unless the user asks for
  a rough dense-LM estimate.
- Including activation-checkpoint recompute in a metric labeled standard MFU.
- Estimating MoE FLOPs from total expert parameters.
- Multiplying routed tokens by `expert_parallel_size`.
- Applying sparse token counts to full-token modules such as router, fusion, or
  LM head.
- Removing all attention FLOPs for KV reuse instead of only reused K/V
  projection FLOPs.
- Counting packed attention as `packed_T^2` when the kernel actually skips
  cross-sample masked pairs.
- Counting only the LLM in a VLM and omitting vision encoder or projector.
- Using text-only sequence length after visual tokens have been inserted.
- Treating all frozen modules as forward-only when gradients still propagate
  through their inputs.
- Mixing forward-only FLOPs with optimizer-step timing unless the metric is
  explicitly labeled as forward-only.
