---
name: fsdp2-prefetching
description: Add a recipe/model-local FSDP2 prefetching callable for new models. Use when
  FSDP2 wrapping already exists but forward or backward prefetch order depends on the
  model's concrete layer layout, branching structure, and forward execution order.
---

# FSDP2 Prefetching

## Goal

- Generate a recipe or model-local FSDP2 prefetch setup callable for the target model.
- Bind that callable on the top-level model class as `APPLY_FSDP2_CUSTOM_PREFETCHING`.
- Keep the implementation local to the recipe or model instead of introducing a generic
  prefetch DSL or runtime helper.

The runtime contract in this repo is fixed:

- Entry point: `mvp_engine/distributed/fsdp2.py`
- Runtime only discovers and calls `model.__class__.APPLY_FSDP2_CUSTOM_PREFETCHING(model)`
  after FSDP2 wrapping
- Do not add a YAML toggle or design a generic prefetch DSL

## Required Inputs

- The target `modeling_*.py` file or equivalent model implementation file
- The top-level model class actually used by training
- The FSDP2 wrap targets or `_no_split_modules`
- The source for the top-level `forward()` and key submodule `forward()` methods
- Whether the model contains branches, cross-layer jumps, mixture layers, or shared blocks

## Workflow

### 1. Decide whether custom prefetching is necessary

- Use this skill mainly when the model architecture is not a simple linear stack, for
  example multi-branch models, cross-layer transitions, mixture layers, or other execution
  paths where default FSDP2 overlap is unlikely to match the real runtime order.
- If the user already sees clear waiting between wrapped modules, branch handoff stalls, or
  poor communication and compute overlap, custom prefetching is a reasonable next step.
- If the model is mostly sequential, default FSDP2 behavior is often sufficient unless the
  user explicitly wants extra performance tuning.

### 2. Collect the structure needed for prefetch wiring

- Find the top-level model class actually used by training.
- Find the repeated compute units actually wrapped by FSDP2, such as encoder layers,
  mixture layers, or heads.
- Only record modules that are wrapped by `fully_shard()`; do not include unwrapped modules
  in prefetch edges.
- Read the top-level `forward()` and key block `forward()` methods and write down the full
  forward execution chain in source order.
- For branched or mixture models, first map the per-layer order inside one stage, then map
  how stages connect to each other.

### 3. Draft the minimum forward and backward prefetch edges

- Forward edge rule:
  - While executing the current module, prefetch the next FSDP2 module that will run
    immediately after it.
  - In branched models, follow the real execution order instead of assuming branches are
    parallel.
- Backward edge rule:
  - Start from the reverse of the forward chain, then add
    `set_modules_to_backward_prefetch()`.
  - Add only the edges that materially reduce waiting; do not connect every adjacent module
    just for completeness.
- For purely sequential stacks, prefer the simplest `layer[i] -> layer[i + 1]` pattern.
- For branch transitions, prefer explicit indices or explicit lists over a generic graph
  algorithm.

### 4. Edit the modeling code

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

- If the modeling file already contains the top-level wrapper class used by training, only
  extend that existing class with `APPLY_FSDP2_CUSTOM_PREFETCHING`; do not create a second
  wrapper class with the same name.
- If the model needs both TP and FSDP2 prefetching, `APPLY_FSDP2_CUSTOM_PREFETCHING`,
  `TP_MODULE_CONFIG`, and `TP_MODULE_POSTPROCESSORS` must be merged onto the same top-level
  model class declaration.
- Keep the callable recipe or model-local; do not move it into `mvp_engine/`.
- The callable should read the already-wrapped module instances directly from `model`; do
  not rebuild shadow module lists elsewhere.
- Use an idempotence guard such as `_fsdp2_prefetching_configured` to avoid double setup.

### 5. Keep the implementation simple

- Do not introduce `torch.fx`, tracing helpers, or automatic graph analysis.
- Do not abstract model-local execution order into a generic runtime helper.
- Do not mutate model config to represent prefetch edges.
- If the wiring is only a few module families, use explicit loops and branches.

## Validation

- Confirm the top-level model class defines `APPLY_FSDP2_CUSTOM_PREFETCHING` and that it
  is callable.
- Confirm that if the top-level wrapper class already existed, this change extended that
  class instead of creating a second class with the same name.
- Confirm that if the model uses both TP and FSDP2 prefetching, the related class
  attributes are merged onto the same top-level model class declaration.
- Confirm the callable resolves real runtime module paths instead of guessed names.
- Confirm every module used in a prefetch edge is part of the FSDP2 wrap set.
- Confirm the callable has an idempotence guard and can be called twice safely.
- Confirm no generic prefetch DSL, graph helper, or YAML config field was introduced.

Add recipe-local assertions under `recipes/<recipe>/tests/skills/fsdp2-prefetching/asserts.py`,
using the standard structure and smoke hooks:

- `tests/test_structure.py`: verify recipe structure and FSDP2 prefetch wiring.
- `tests/test_smoke.py`: run one real recipe-owned training step and checkpoint/log path.
- `tests/skills/fsdp2-prefetching/test_prefetch_edges_impact.py`: create a recipe-local impact test that uses
  `mvp_engine.test.recipe_probe` helpers, then verify the real FSDP2-wrapped
  modules hold the expected prefetch edges. Resolve each expected module path to
  the live module, read its FSDP2 state, and compare forward/backward prefetch
  targets by object identity against the next modules in the recipe's real
  execution order. This test should prove the hook installed the intended
  runtime edges, not that loss or throughput changed.

## Output

- State which model file was added or updated and which
  `APPLY_FSDP2_CUSTOM_PREFETCHING` callable was bound.
- Summarize the core forward and backward prefetch edges.
- State what was validated and what remains unverified.

## Read On Demand

- Read `./references/vit_classification/model/vit.py` when you need a sequential-stack
  FSDP2 prefetching reference.
