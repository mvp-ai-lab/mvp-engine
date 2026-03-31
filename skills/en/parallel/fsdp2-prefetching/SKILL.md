---
name: fsdp2-prefetching
description: Add a recipe/model-local FSDP2 prefetching callable for new models. Use this when FSDP2 wrapping already exists but forward/backward prefetch order depends on the model's concrete layer layout, branching structure, and forward execution order.
---

# FSDP2 Prefetching (EN)

## Goal

Generate a recipe/model-local FSDP2 prefetch setup callable for the target model and bind it on the top-level model class as `APPLY_FSDP2_CUSTOM_PREFETCHING`.

The runtime contract in this repo is fixed:
- Entry point: `mvp_engine/distributed/parallelize.py`
- Runtime only discovers and calls `model.__class__.APPLY_FSDP2_CUSTOM_PREFETCHING(model)` after FSDP2 wrapping
- Do not add a YAML toggle or design a generic prefetch DSL

## Required Inputs

- Target `modeling_*.py` or equivalent model implementation file
- Top-level model class used by training
- FSDP2 wrap targets or `_no_split_modules`
- Source for the top-level `forward()` and key submodule `forward()` methods
- Whether the model contains branches, cross-layer jumps, mixture layers, or shared blocks

## When To Use

- Use this skill when the model architecture is fairly complex, such as multi-branch stacks, cross-layer transitions, mixture layers, or any forward path that is not just a simple linear stack. In those cases, a custom FSDP2 prefetching policy can better match the true execution order and improve training throughput.
- Use it when the user already sees signs that default FSDP2 overlap is not good enough, for example obvious waiting between wrapped modules, branch handoff stalls, or poor communication/compute overlap.
- If the model is very linear and mostly a standard sequential stack, you often do not need custom prefetching at all. The default FSDP2 behavior is usually enough unless the user explicitly wants extra performance tuning.
- When executing this skill, explain this tradeoff to the user first: this skill is mainly for models where default prefetch behavior does not match the real execution topology, not something every FSDP2 model must have.

## Workflow

### 1. Collect the structure needed for prefetch wiring

- Find the top-level model class actually used by training.
- Find the repeated compute units actually wrapped by FSDP2, such as encoder layers, mixture layers, or heads.
- Only record modules that are wrapped by `fully_shard()`; do not include unwrapped modules in prefetch edges.
- Read the top-level `forward()` and key block `forward()` methods and write down the full forward execution chain in source order.
- For branched or mixture models, first map the per-layer order inside one stage, then map how stages connect to each other.

### 2. Draft the minimum forward/backward prefetch edges

- Forward edge rule:
  - While executing the current module, prefetch the next FSDP2 module that will run immediately after it.
  - In branched models, follow the real execution order instead of assuming branches are parallel.
- Backward edge rule:
  - Start from the reverse of the forward chain, then add `set_modules_to_backward_prefetch()`.
  - Add only the edges that materially reduce waiting; do not connect every adjacent module just for completeness.
- For purely sequential stacks, prefer the simplest layer[i] -> layer[i+1] pattern.
- For branch transitions, prefer explicit indices or explicit lists over a generic graph algorithm.

### 3. Edit the modeling code

- Add a minimal callable in the modeling file, for example:
  ```python
  def apply_fsdp2_custom_prefetching_for_<model_name>(model: nn.Module) -> None:
      if getattr(model, "_fsdp2_prefetching_configured", False):
          return
      ...
      layer_a.set_modules_to_forward_prefetch([layer_b])
      layer_b.set_modules_to_backward_prefetch([layer_a])
      model._fsdp2_prefetching_configured = True
  ```
- Then bind it on the top-level model class:
  ```python
  class <TopModelClass>(...):
      APPLY_FSDP2_CUSTOM_PREFETCHING = apply_fsdp2_custom_prefetching_for_<model_name>
  ```
- If the modeling file already contains the top-level wrapper class used by training, only extend that existing class with `APPLY_FSDP2_CUSTOM_PREFETCHING`; do not create a second wrapper class with the same name.
- If the model needs both TP and FSDP2 prefetching, `APPLY_FSDP2_CUSTOM_PREFETCHING`, `TP_MODULE_CONFIG`, and `TP_MODULE_POSTPROCESSORS` must be merged onto the same top-level model class declaration.
- Keep the callable recipe/model-local; do not move it into `mvp_engine/`.
- The callable should read the already-wrapped module instances directly from `model`; do not rebuild shadow module lists elsewhere.
- Use an idempotence guard such as `_fsdp2_prefetching_configured` to avoid double setup.

### 4. Keep the implementation simple

- Do not introduce `torch.fx`, tracing helpers, or automatic graph analysis.
- Do not abstract model-local execution order into a generic runtime helper.
- Do not mutate model config to represent prefetch edges.
- If the wiring is only a few module families, use explicit loops and branches.

## Validation

- Confirm the top-level model class defines `APPLY_FSDP2_CUSTOM_PREFETCHING` and that it is callable.
- Confirm that if the top-level wrapper class already existed, this change extended that class instead of creating a second class with the same name.
- Confirm that if the model uses both TP and FSDP2 prefetching, the related class attributes are merged onto the same top-level model class declaration.
- Confirm the callable resolves real runtime module paths instead of guessed names.
- Confirm every module used in a prefetch edge is part of the FSDP2 wrap set.
- Confirm the callable has an idempotence guard and can be called twice safely.
- Confirm no generic prefetch DSL, graph helper, or YAML config field was introduced.

## Output

- State which model file was added or updated and which `APPLY_FSDP2_CUSTOM_PREFETCHING` callable was bound.
- Summarize the core forward/backward prefetch edges.
- State what was validated and what remains unverified.

## Read On Demand

- When you need a sequential-stack FSDP2 prefetching reference, read `./references/vit_classification/model/vit.py`.
