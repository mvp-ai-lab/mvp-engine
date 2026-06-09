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
single `apply_pre_build(...)`:

- **official**: dispatch to liger's own `apply_liger_kernel_to_<family>`, with the
  family inferred from Hugging Face `AutoConfig.model_type` by default;
- **custom** (model has no official helper): apply an explicit `{module:
  LigerPatch}` map describing the same symbol swaps for the model's own modeling
  module.

There is no post-build / instance path: all patching is pre-build, so no
module-tree walking is needed. Loss kernels stay off unless explicitly allowed.

## Required Inputs

- model name/path (official route) or a model-family label for reporting;
- module selection: `"auto"` or `dict[str, bool]`;
- for custom models: a `custom_patches` map (`{semantic_module: LigerPatch}`);
- whether loss kernels are compatible with the recipe's loss accounting.

## Workflow

### 1. Initialize The Kit

```python
from mvp_engine.kit import LigerKernelKit

self.liger_kit = LigerKernelKit()
```

`liger-kernel` is an optional dependency. Importing the kit must not require it;
the package is imported lazily only when `apply_pre_build` runs the official route.

### 2. Official Route (model is in liger's registry)

Call before model construction:

```python
report = self.liger_kit.apply_pre_build(
    model_name_or_path=config.model.pretrained_model_name_or_path,
    modules=config.model.liger_kernel.modules,            # "auto" or {flag: bool}
    model_family=config.model.liger_kernel.get("model_family_override"),
)
model = build_model(...)
```

- `modules="auto"` forwards **nothing** for rope/norm/mlp (liger picks the correct
  per-model defaults, e.g. SwiGLU vs GeGLU, standard vs multimodal RoPE) and only
  forces loss kernels **off**. This avoids re-encoding per-model knowledge the
  library already owns.
- An explicit `dict` is forwarded as-is; enabling a module the helper does not
  accept fails fast. Accepted flags are trusted verbatim — liger may no-op some on
  a given model (e.g. dense Qwen3-VL SwiGLU), so prefer `auto` to defer to liger's
  correct per-model defaults.
- Use `model_family` only to override a custom or misreported `model_type`.

### 3. Custom Route (no official helper)

Provide the symbol swaps as data; the kit imports each target module, checks the
symbol exists (catching upstream renames), and `setattr`s the replacement:

```python
from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from mvp_engine.kit import LigerPatch

M = "my_pkg.modeling_mymodel"
report = self.liger_kit.apply_pre_build(
    model_family="mymodel",
    custom_patches={
        "rms_norm": LigerPatch(module=M, attr="MyRMSNorm", replacement=LigerRMSNorm),
        "swiglu":   LigerPatch(module=M, attr="MyMLP", replacement=LigerSwiGLUMLP),
        "rope":     LigerPatch(module=M, attr="apply_rotary_pos_emb", replacement=liger_rotary_pos_emb),
    },
)
model = build_model(...)
```

The kit does **not** infer which symbols to swap or which replacement is correct —
that per-model knowledge is authored via `skills/model/liger-kernel/SKILL.md`. The
kit only provides the validated mechanism. For composite custom models, prefer
reusing the official route per component (see that skill).

### 4. Semantic Module Names

```text
rope  rms_norm  layer_norm  swiglu  geglu  cross_entropy  fused_linear_cross_entropy
```

Unknown names are rejected. In the custom route, `modules="auto"` applies every
provided patch; an explicit `dict` applies only enabled flags and fails if an
enabled flag has no patch.

### 5. Loss Kernels

`cross_entropy` and `fused_linear_cross_entropy` are disabled by default because
many recipes own loss reduction or token normalization (liger defaults FLCE to
**on**, which the kit overrides off). Set `loss_kernels_allowed=True` only after
the recipe preserves the expected loss contract. The custom route applies
module-level symbol swaps only; `fused_linear_cross_entropy` rewrites a model's
`ForCausalLM.forward` and is **out of scope** for custom models here.

## Validation

### Soft Validation

- official route infers from `AutoConfig.model_type` unless overridden;
- `liger-kernel` remains optional and lazily imported;
- enabling a module unsupported by the official helper fails clearly;
- custom patches fail clearly when a target symbol is missing;
- custom per-model knowledge lives in the recipe, not the kit;
- loss kernels are guarded by recipe compatibility.

### Hard Validation

```bash
pytest tests/test_liger_kernel_kit.py -q
```

For recipe usage, also run the recipe structure test and a smoke test in an
environment with `liger-kernel` and the required accelerator resources.

## Output

- State route (official/custom), resolved modules, and helper or patched symbols
  (from the returned `LigerKernelReport`).
- State whether loss kernels are allowed.
- Report validation commands and runtime gaps.

## Read On Demand

- `skills/model/liger-kernel/SKILL.md`: recipe placement and how to author a
  custom-model (or composite-model) `custom_patches` map.
