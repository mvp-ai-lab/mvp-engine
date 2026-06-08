---
name: liger-kernel-kit
description: Use LigerKernelKit for reusable Liger Kernel integration,
  including official pre-build model-family patching, post-build module
  replacement, module selection validation, optional recipe-specific replacers,
  and loss-kernel guards.
---

# Liger Kernel Kit

## Goal

Use `LigerKernelKit` as the default API for Liger Kernel integration:

- `apply_pre_build(...)` calls official `liger_kernel.transformers` family
  patch functions before model construction, inferring the family from Hugging
  Face `AutoConfig.model_type` by default;
- `apply_post_build(...)` replaces supported modules on an already-built model;
- `resolve_modules(...)` validates semantic module selections;
- built-in post-build replacement covers generic norm modules;
- optional recipe-specific replacers handle model-family details.

## Required Inputs

- model name/path for pre-build or a loaded model for post-build;
- optional model family override, such as `qwen3`, `qwen3_vl`, or `llama`;
- stage: `pre_build` or `post_build`;
- module selection: `"auto"` or `dict[str, bool]`;
- whether loss kernels are compatible with the recipe's loss accounting;
- optional recipe-specific replacers for modules the kit cannot handle
  generically.

## Workflow

### 1. Initialize The Kit

```python
from mvp_engine.kit import LigerKernelKit

self.liger_kit = LigerKernelKit()
```

`liger-kernel` is an optional dependency. Importing the kit must not require it;
the package is imported lazily only when an apply method runs.

### 2. Apply Pre-Build Family Patches

Use pre-build when Liger provides an official model-family patch:

```python
if config.model.liger_kernel.enabled and config.model.liger_kernel.stage == "pre_build":
    self.liger_kit.apply_pre_build(
        model_name_or_path=config.model.pretrained_model_name_or_path,
        modules=config.model.liger_kernel.modules,
        model_family=config.model.liger_kernel.get("model_family_override"),
    )
model = build_model(...)
```

The kit forwards explicit `False` values for accepted kwargs, so Liger defaults
cannot silently enable kernels the recipe left disabled.
Use `model_family` only as an override for custom or misreported configs.

### 3. Apply Post-Build Replacements

Use post-build after model construction and before freeze, compile, and
distributed wrapping:

```python
model = self.liger_kit.apply_post_build(
    model,
    modules=config.model.liger_kernel.modules,
    model_family=config.model.liger_kernel.get("model_family_override"),
)
```

Built-in post-build support is intentionally narrow:

```text
rms_norm
layer_norm
```

For other modules, pass explicit recipe-specific replacers:

```python
model = self.liger_kit.apply_post_build(
    model,
    modules={"swiglu": True},
    model_family="custom_family",
    module_replacers={"swiglu": replace_custom_swiglu},
)
```

A replacer receives `(model, liger_transformers)` and returns
`LigerReplacement` records or equivalent dictionaries with `path`, `source`, and
`target`.

### 4. Resolve Modules Conservatively

Use semantic module names:

```text
rms_norm
layer_norm
rope
swiglu
geglu
cross_entropy
fused_linear_cross_entropy
```

`modules="auto"` enables only the kit's conservative defaults for the selected
stage and model family. Explicit unsupported modules fail in strict mode.

### 5. Handle Loss Kernels Explicitly

`cross_entropy` and `fused_linear_cross_entropy` are disabled by default because
many recipes own loss reduction or token normalization. Set
`loss_kernels_allowed=True` only after the recipe preserves the expected loss
contract.

## Validation

### Soft Validation

- pre-build infers from `AutoConfig.model_type` unless an override is provided;
- post-build infers from `model.config.model_type` unless an override is provided;
- no recipe duplicates kit helper logic;
- `liger-kernel` remains optional and lazily imported;
- unsupported model-family/module combinations fail clearly;
- post-build replacement preserves parameter names, dtype, device, and
  `requires_grad`;
- custom replacers are recipe-local and explicit;
- loss kernels are guarded by recipe compatibility.

### Hard Validation

Run focused kit tests:

```bash
pytest tests/test_liger_kernel_kit.py -q
```

For recipe usage, also run the recipe structure test and a smoke test in an
environment with `liger-kernel` and the required accelerator resources.

## Output

- State selected stage and resolved modules.
- State official helper or post-build replacers used.
- State whether loss kernels are allowed.
- Report validation commands and runtime gaps.

## Read On Demand

- `skills/model/liger-kernel/SKILL.md`: recipe placement workflow.
