---
name: liger-kernel-kit
description: Use LigerKernelKit for reusable Liger Kernel integration before model
  construction, covering official model-family dispatch, custom-model symbol
  patching via LigerPatch, module selection validation, and loss-kernel guards.
---

# Liger Kernel Kit

## Goal

Use `LigerKernelKit` as the single API for Liger Kernel integration. Every Liger
helper (`apply_liger_kernel_to_<family>`) is the same skeleton: flag-gated
`setattr(modeling_module, symbol, liger_impl)` assignments run **before** the
model is built. The kit keeps that one skeleton and exposes two routes through a
single `apply(...)`:

- **official**: dispatch to liger's own `apply_liger_kernel_to_<family>`, with the
  family inferred from Hugging Face `AutoConfig.model_type` by default;
- **custom** (model has no official helper): apply an explicit `{module:
  LigerPatch}` map describing the same symbol swaps for the model's own modeling
  module.

There is no instance-patching path: everything runs before the model is built,
so no module-tree walking is needed. `excluded_modules` disables any kernels and
defaults to the loss kernels.

## Required Inputs

- model name/path (official route) or a model-family label for reporting;
- module selection: `"auto"` or `dict[str, bool]`;
- for custom models: a `custom_patches` map (`{module_name: LigerPatch}`);
- which modules to exclude (defaults to the loss kernels).

## Workflow

### 1. Initialize The Kit

```python
from mvp_engine.kit import LigerKernelKit

self.liger_kit = LigerKernelKit()
```

`liger-kernel` is an optional dependency. Importing the kit must not require it;
the package is imported lazily only when `apply` runs the official route.

### 2. Official Route (model is in liger's registry)

Call before model construction (recipe config wiring lives in
`skills/model/liger-kernel/SKILL.md`):

```python
report = self.liger_kit.apply(model_name_or_path=..., modules="auto")
model = build_model(...)
```

- `modules="auto"` forwards **nothing** for rope/norm/mlp (liger picks the correct
  per-model defaults, e.g. SwiGLU vs GeGLU, standard vs multimodal RoPE) and only
  forces the `excluded_modules` **off**. This avoids re-encoding per-model
  knowledge the library already owns.
- An explicit `dict` is forwarded as-is; enabling a module the helper does not
  accept fails fast. Accepted flags are trusted verbatim — liger may no-op some on
  a given model (e.g. dense Qwen3-VL SwiGLU), so prefer `auto` to defer to liger's
  correct per-model defaults.
- Use `model_family` only to override a custom or misreported `model_type`. It is
  used verbatim — match liger's naming (e.g. `qwen3_vl`, not `Qwen3-VL`).

### 3. Custom Route (no official helper)

Provide the symbol swaps as data; the kit imports each target module, fails if a
symbol is missing (catching upstream renames), `setattr`s the replacement, and
reports the patched paths:

```python
from liger_kernel.transformers import LigerRMSNorm

from mvp_engine.kit import LigerKernelKit

liger_kit = LigerKernelKit()

report = liger_kit.apply(
    model_family="mymodel",
    custom_patches={"rms_norm": liger_kit.Patch("my_pkg.modeling_mymodel", "MyRMSNorm", LigerRMSNorm)},
)
model = build_model(...)
```

The kit does **not** infer which symbols to swap or which replacement is
numerically correct. Authoring that map — symbol discovery, the replacement
decision table, composite/vendored models — is `skills/model/liger-kernel/SKILL.md`.

### 4. Semantic Module Names

```text
rope  rms_norm  layer_norm  swiglu  geglu  cross_entropy  fused_linear_cross_entropy
```

Unknown names are rejected on the **official route only** — its vocabulary is the
set of kwargs liger's helpers accept. The custom route accepts **any** module
name, so a custom model can cover kernels beyond this list (e.g. a softmax or
embedding swap); the recipe owns the map's contents. `modules` selects modules on
the official route only.

### 5. Excluded Modules

`excluded_modules` disables the named kernels on either route (forced off on the
official route, skipped on the custom route). It defaults to the loss kernels
(`cross_entropy`, `fused_linear_cross_entropy`) because many recipes own loss
reduction or token normalization and liger defaults FLCE to **on**. Pass
`excluded_modules=()` only after the recipe preserves the expected loss contract.
Note `fused_linear_cross_entropy` rewrites a model's `ForCausalLM.forward`, so it
cannot be expressed as a custom symbol swap anyway.

## Validation

### Soft Validation

- the kit is called before the model is built;
- custom per-model knowledge lives in the recipe, not the kit;
- `excluded_modules` keeps the loss kernels off unless the recipe owns a
  compatible loss path.

### Hard Validation

There are no kit-level test files. Validate at the recipe level: copy
`skills/model/liger-kernel/references/asserts.py` to
`recipes/<recipe>/tests/skills/liger-kernel/asserts.py` so the recipe structure
and smoke tests verify the wiring and the runtime replacement, and compare
early-step smoke loss with Liger on/off on GPU/NPU.

## Output

- State route (official/custom), resolved modules, and helper or patched symbols
  (from the returned `LigerKernelReport`).
- State the excluded modules.
- Report validation commands and runtime gaps.

## Read On Demand

- `skills/model/liger-kernel/SKILL.md`: recipe placement and how to author a
  custom-model (or composite-model) `custom_patches` map.
