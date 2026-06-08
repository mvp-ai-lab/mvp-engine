---
name: liger-kernel
description: Decide where and how to wire Liger Kernel into an MVP-Engine recipe
  using LigerKernelKit, including pre-build family patch placement, post-build
  model patch placement, module selection, and loss-kernel compatibility.
---

# Liger Kernel

## Goal

Use Liger Kernel to replace supported model kernels without changing training
semantics. Treat reusable behavior as `LigerKernelKit` API and keep recipes
limited to config, placement, and any model-family-specific extension points.

## Required Inputs

- target recipe and model construction path;
- model name/path for pre-build or a loaded model for post-build;
- optional model family override, such as `qwen3`, `qwen3_vl`, or `llama`;
- desired stage: `pre_build` or `post_build`;
- module selection, either `"auto"` or explicit semantic module flags;
- whether the recipe has custom loss accounting that conflicts with Liger loss
  kernels.

## Workflow

### 1. Use LigerKernelKit

Read `skills/kit/liger-kernel-kit/SKILL.md` before editing. Do not reimplement
module resolution, Liger imports, pre-build helper dispatch, or generic norm
replacement inside a recipe.

### 2. Choose The Stage

Use `pre_build` when Liger provides an official model-family patch function.
Call the kit before model construction:

```python
self.liger_kit.apply_pre_build(
    model_name_or_path=config.model.pretrained_model_name_or_path,
    modules=config.model.liger_kernel.modules,
    model_family=config.model.liger_kernel.get("model_family_override"),
)
model = self.model_kit.build_model(...)
```

Use `post_build` when a recipe needs instance-level module replacement. Route it
through the recipe's existing model patch stage:

```python
model_patches = [...]
if config.model.liger_kernel.enabled and config.model.liger_kernel.stage == "post_build":
    model_patches.append(
        partial(
            self.liger_kit.apply_post_build,
            modules=config.model.liger_kernel.modules,
            model_family=config.model.liger_kernel.get("model_family_override"),
            module_replacers=recipe_replacers,
        )
    )
model = self.model_kit.apply_model_patches(model, model_patches)
```

If the recipe uses a non-MLLM model kit, place post-build Liger at the equivalent
post-construction, pre-freeze, pre-distributed wrapping point.

### 3. Keep Recipe Glue Small

Recipe code may add:

- config fields under `model.liger_kernel`;
- a `LigerKernelKit` instance in the engine;
- optional model-family override wiring;
- recipe-specific `module_replacers` only when the kit does not support the
  module generically.

Do not apply Liger to a recipe unless requested for that recipe. This skill can
exist independently from recipe usage.

### 4. Protect Loss Semantics

Leave `cross_entropy` and `fused_linear_cross_entropy` disabled unless the
recipe has a dedicated compatibility path for its loss accounting. This matters
for token-normalized or unreduced per-token loss workflows.

## Validation

### Soft Validation

- recipe calls `LigerKernelKit` rather than local duplicate helper code;
- pre-build runs before model construction;
- post-build runs after model construction and before freeze, compile, and
  distributed wrapping;
- unsupported modules fail clearly instead of being reported as applied;
- loss kernels are not silently enabled when recipe loss accounting is custom.

### Hard Validation

Run the Liger kit unit tests and the target recipe's structure test. Run smoke
with Liger enabled only in an environment that has `liger-kernel` installed and
the required GPU/NPU resources.

## Output

- State selected stage and module selection.
- State whether kit built-ins or recipe-specific replacers were used.
- State validation commands and any runtime environment gap.

## Read On Demand

- `skills/kit/liger-kernel-kit/SKILL.md`: authoritative API contract.
- `skills/kit/mllm-model-kit/SKILL.md`: MLLM model setup placement.
- `skills/kit/token-loss-kit/SKILL.md`: token-loss compatibility.
