---
name: liger-kernel
description: Decide where and how to wire Liger Kernel into an MVP-Engine recipe
  using LigerKernelKit, including official family dispatch, authoring a custom or
  composite model's LigerPatch map, module selection, and loss-kernel safety.
---

# Liger Kernel

## Goal

Use Liger Kernel to replace supported model kernels **before model construction**
without changing training semantics. Reusable behavior is the `LigerKernelKit`
API; the recipe owns only config, placement, and — for custom models — the
per-model `custom_patches` map. All application happens before the model is
built (no instance patching).

## Required Inputs

- target recipe and its model construction point;
- model name/path (official) or the custom model's modeling module + symbol names;
- optional model-family override (e.g. `qwen3`, `qwen3_vl`);
- module selection: `"auto"` or explicit semantic flags;
- whether the recipe's loss accounting conflicts with Liger loss kernels.

## Workflow

### 1. Use LigerKernelKit

Read `skills/kit/liger-kernel-kit/SKILL.md` first. Do not reimplement module
resolution, Liger imports, official dispatch, or symbol patching in a recipe.

### 2. Classify The Model, Then Pick A Route

Decide once, before model construction:

1. **Whole model in [liger's registry](https://github.com/linkedin/Liger-Kernel)**
   (`llama`, `qwen2/3`, `qwen2_5_vl`, `gemma*`, `glm4v`, `llava`, `mllama`, ...) →
   **official route**, one call. Done.
2. **Composite custom model** = a known LLM family + a custom encoder/projector
   (e.g. an OV2-style VLM whose LLM is Qwen3) → **reuse official per component**
   for the known family, then a small `custom_patches` map for the novel parts.
3. **Monolithic custom model** with no official helper → author a `custom_patches`
   map for its own modeling module.

Always call the kit *before* the model is built:

```python
self.liger_kit.apply(
    model_name_or_path=config.model.pretrained_model_name_or_path,
    modules=config.model.liger_kernel.modules,
    model_family=config.model.liger_kernel.get("model_family_override"),
)
model = self.model_kit.build_model(...)
```

### 3. Author A Custom / Composite `custom_patches` Map

This is the per-model knowledge that cannot be generic. To produce it:

1. **Locate the modeling module** the model is built from (a shared
   `transformers.models.<x>.modeling_<x>`, or the recipe-local / remote-code
   modeling file). It must be importable before `build_model`.
2. **Find the symbol names** to swap:
   `grep -nE "class .*RMSNorm|class .*MLP|def apply_.*rotary" <modeling_file>`.
3. **Choose the matching Liger replacement** using the table below — picking the
   wrong one silently corrupts the model.
4. Pass them as `liger_kit.Patch(module, attr, replacement)` entries.

| Module | Source looks like | Correct Liger replacement |
|---|---|---|
| `rms_norm` | standard RMSNorm (`x * rsqrt(mean(x²)+eps) * weight`) | `LigerRMSNorm` |
| `rms_norm` | **Gemma-style** (`(1+weight)`, float32 upcast / offset) | `LigerRMSNormForGemma` — **never** the standard one |
| `layer_norm` | `nn.LayerNorm` | `LigerLayerNorm` |
| `swiglu` | gate/up/down MLP with SiLU | `LigerSwiGLUMLP` |
| `geglu` | gate/up/down MLP with GELU | `LigerGEGLUMLP` |
| `rope` | standard full rotary | `liger_rotary_pos_emb` |
| `rope` | **multimodal / mRoPE** (VL 3D positions) | not `liger_rotary_pos_emb`; only the matching multimodal rope, else leave off |

For a **composite** model, reuse the official route for the known family instead
of re-listing its symbols, then patch only the novel component:

```python
# LLM backbone is shared transformers Qwen3 -> reuse official (full kernel set)
self.liger_kit.apply(model_family="qwen3", modules="auto")
# custom vision encoder -> only norms it actually uses
self.liger_kit.apply(
    model_family="myvlm",
    custom_patches={
        "rms_norm": self.liger_kit.Patch("my_pkg.modeling_encoder", "EncoderRMSNorm", LigerRMSNorm)
    },
)
model = self.model_kit.build_model(...)
```

If the LLM backbone is a **vendored** copy (not shared `transformers`), the
official call cannot reach it — give the backbone its own `custom_patches`
pointing at the vendored modeling module.

### 4. Keep Recipe Glue Small

Recipe code may add: config under `model.liger_kernel`; a `LigerKernelKit`
instance; an optional family override; and — only for custom models — the
`custom_patches` map. Do not apply Liger to a recipe unless requested for it.

### 5. Protect Loss Semantics

Leave `cross_entropy` and `fused_linear_cross_entropy` off unless the recipe has
a dedicated loss-compatibility path (matters for token-normalized / unreduced
per-token loss). The default `excluded_modules` already keeps them off; the
mechanics are kit contract — see the kit skill.

## Validation

### Soft Validation

- recipe calls `LigerKernelKit`, not duplicate helper code;
- all application happens before model construction;
- the chosen Liger replacement matches the source module's math (table above);
- enabling an unsupported module fails clearly instead of being reported applied;
- the default `excluded_modules` (loss kernels) stays in place under custom loss
  accounting.

### Hard Validation

Copy `references/asserts.py` (next to this skill) to
`recipes/<recipe>/tests/skills/liger-kernel/asserts.py` so the recipe structure
and smoke tests verify the wiring and the runtime replacement. Liger kernels are
Triton kernels that run on GPU and NPU; **numerical correctness is validated
empirically by smoke** (compare the first training steps' loss with and without
Liger), not at patch time.

## Output

- State the route, module selection, and helper or patched symbols.
- State whether kit official dispatch or a custom map was used.
- State validation commands and any runtime environment gap.

## Read On Demand

- `skills/kit/liger-kernel-kit/SKILL.md`: authoritative API contract.
- `skills/kit/mllm-model-kit/SKILL.md`: MLLM model setup placement.
- `skills/kit/token-loss-kit/SKILL.md`: token-loss compatibility.
