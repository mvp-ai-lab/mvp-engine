# MoE and Sparse Routing MFU

## Use When

Use for Mixture-of-Experts models, expert parallel training, top-k routing,
shared experts, routed FFN blocks, MoD/token routing, selected-token compute,
KV reuse, or sparse branches.

## Counting Rules

- Count activated expert FLOPs, not total expert parameter FLOPs.
- Prefer router-produced per-expert token counts.
- For top-k routing, each token contributes to each selected expert path.
- Count router/gate linear FLOPs separately when material.
- Count shared dense experts for all tokens that pass through them.
- Do not count all-to-all, dispatch, combine, padding movement, or communication as model FLOPs.
- Count dropped tokens only for the compute they actually execute.
- Count padded expert capacity only if the implementation performs expert
  compute on padded slots.

## Sparse Routing Rules

- For MoD or token routing, use selected token count `k` for routed attention,
  routed FFN, or sparse branch compute.
- Use full token count `T` for router scoring, token selection, fusion, residual
  projections, final output layers, and LM head unless those modules are also
  sparse.
- For KV reuse, remove K/V projection FLOPs for layers that reuse external KV.
  Keep Q projection, QK, AV, output projection, and any fusion compute.
- For shared dense paths plus sparse branches, count both paths for the tokens
  that execute them.
- For top-k sparse branches, multiply branch compute by the number of selected
  branches per token, or use observed per-branch token counts when available.

## Distributed Rules

- For global MFU, count logical global routed tokens across data-parallel samples.
- Do not multiply tokens by expert_parallel_size unless EP also changes the number of input samples.
- Expert parallelism changes ownership of experts, not the logical amount of model compute.
- Communication overhead lowers MFU through measured step time.
- For expert parallel, count all activated experts logically, including experts
  owned by other ranks.
- Do not report rank-local routed FLOPs as global MFU unless the denominator is
  also rank-local and the metric is labeled separately.

## Fallbacks

- If per-expert token counts are available, use them.
- If unavailable, estimate activated expert tokens as `tokens * top_k`, and label/report the estimate.
- If routing drops tokens, count only tokens that actually execute expert compute.
- If sparse branch token counts are unavailable, use the configured selected
  token count or routing ratio, and label/report the estimate.

## Common Mistakes

- Estimating MoE FLOPs from total expert parameters.
- Multiplying routed tokens by `expert_parallel_size`.
- Counting expert all-to-all as model FLOPs.
- Ignoring shared experts or dense fallback paths.
- Applying sparse token count to full-token modules such as router, fusion, or
  LM head.
- Removing all attention FLOPs for KV reuse instead of only reused K/V
  projection FLOPs.
