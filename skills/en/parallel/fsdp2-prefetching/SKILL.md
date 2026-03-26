---
name: fsdp2-prefetching
description: Add a recipe/model-local FSDP2 prefetching callable for new models. Use this when FSDP2 wrapping already exists but forward/backward prefetch order depends on the model's concrete layer layout, branching structure, and forward execution order.
---

# FSDP2 Prefetching (EN)

## Goal

Generate a recipe/model-local FSDP2 prefetch setup callable for the target model and bind it on the top-level model class as `FSDP2_PREFETCHING`.

The runtime contract in this repo is fixed:
- Entry point: `mvp_engine/distributed/parallelize.py`
- Runtime only discovers and calls `model.__class__.FSDP2_PREFETCHING(model)` after FSDP2 wrapping
- Do not add a YAML toggle or design a generic prefetch DSL

## Required Inputs

- Target `modeling_*.py` or equivalent model implementation file
- Top-level model class used by training
- FSDP2 wrap targets or `_no_split_modules`
- Source for the top-level `forward()` and key submodule `forward()` methods
- Whether the model contains branches, cross-layer jumps, mixture layers, or shared blocks

## Workflow

### 1. Decide whether this should be a skill

- Use a skill when prefetch order depends on model topology, branch transitions, or custom execution order.
- If the behavior is a single generic runtime option, it belongs in code instead.
- Do not invent a generic configuration language for prefetching.

### 2. Collect the structure needed for prefetch wiring

- Find the top-level model class actually used by training.
- Find the repeated compute units actually wrapped by FSDP2, such as encoder layers, mixture layers, or heads.
- Only record modules that are wrapped by `fully_shard()`; do not include unwrapped modules in prefetch edges.
- Read the top-level `forward()` and key block `forward()` methods and write down the full forward execution chain in source order.
- For branched or mixture models, first map the per-layer order inside one stage, then map how stages connect to each other.

### 3. Draft the minimum forward/backward prefetch edges

- Forward edge rule:
  - While executing the current module, prefetch the next FSDP2 module that will run immediately after it.
  - In branched models, follow the real execution order instead of assuming branches are parallel.
- Backward edge rule:
  - Start from the reverse of the forward chain, then add `set_modules_to_backward_prefetch()`.
  - Add only the edges that materially reduce waiting; do not connect every adjacent module just for completeness.
- For purely sequential stacks, prefer the simplest layer[i] -> layer[i+1] pattern.
- For branch transitions, prefer explicit indices or explicit lists over a generic graph algorithm.

### 4. Edit the modeling code

- Add a minimal callable in the modeling file, for example:
  ```python
  def setup_<model_name>_fsdp2_prefetching(model: nn.Module) -> None:
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
      FSDP2_PREFETCHING = setup_<model_name>_fsdp2_prefetching
  ```
- If the modeling file already contains the top-level wrapper class used by training, only extend that existing class with `FSDP2_PREFETCHING`; do not create a second wrapper class with the same name.
- If the model needs both TP and FSDP2 prefetching, `FSDP2_PREFETCHING`, `TP_MODULE_CONFIG`, and `TP_MODULE_POSTPROCESSORS` must be merged onto the same top-level model class declaration.
- Keep the callable recipe/model-local; do not move it into `mvp_engine/`.
- The callable should read the already-wrapped module instances directly from `model`; do not rebuild shadow module lists elsewhere.
- Use an idempotence guard such as `_fsdp2_prefetching_configured` to avoid double setup.

### 5. Keep the implementation simple

- Do not introduce `torch.fx`, tracing helpers, or automatic graph analysis.
- Do not abstract model-local execution order into a generic runtime helper.
- Do not mutate model config to represent prefetch edges.
- If the wiring is only a few module families, use explicit loops and branches.

## Validation

- Confirm the top-level model class defines `FSDP2_PREFETCHING` and that it is callable.
- Confirm that if the top-level wrapper class already existed, this change extended that class instead of creating a second class with the same name.
- Confirm that if the model uses both TP and FSDP2 prefetching, the related class attributes are merged onto the same top-level model class declaration.
- Confirm the callable resolves real runtime module paths instead of guessed names.
- Confirm every module used in a prefetch edge is part of the FSDP2 wrap set.
- Confirm the callable has an idempotence guard and can be called twice safely.
- Confirm no generic prefetch DSL, graph helper, or YAML config field was introduced.
- Add at least one test:
  - lightweight unit test that the callable is invoked by runtime, or
  - smoke test that the parallelized model completes one forward/backward pass

## Output

- State which model file was added or updated and which `FSDP2_PREFETCHING` callable was bound.
- Summarize the core forward/backward prefetch edges.
- State what was validated and what remains unverified.

## Read On Demand

- When you need a sequential-stack FSDP2 prefetching reference, read `./references/vit_classification/model/vit.py`.
