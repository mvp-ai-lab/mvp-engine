---
name: liger-kernel
description: Add, review, update, and validate Liger Kernel integration in
  MVP-Engine recipes, including official pre-build model-family patching,
  recipe-local post-build module replacement, module selection config, and
  token-loss compatibility checks.
---

# Liger Kernel

## Goal

Wire Liger Kernel into a recipe without changing training semantics:

- expose a recipe-local `model.liger_kernel` config;
- support official pre-build model-family patching before `build_model(...)`;
- support recipe-local post-build module replacement through
  `MLLMModelKit.apply_model_patches(...)`;
- let users select modules explicitly, with `"auto"` as the default;
- keep loss-kernel changes separate from token-normalized loss.

## Required Inputs

Identify these before editing:

- target recipe path and `prepare_model()` path;
- model family from config, checkpoint path, or loaded config;
- whether Liger has an official `apply_liger_kernel_to_*` API for that family;
- requested stage: `pre_build` or `post_build`;
- requested modules, or whether `modules="auto"` should be resolved;
- whether the recipe applies `TokenNormedLossKit.apply_chunked_token_loss_patch`.

Ask the user only when the target recipe or intended stage cannot be derived.

## Workflow

### 1. Add Recipe Config

Add a recipe-local config nested under `model`:

```python
class LigerKernelConfig(BaseModel):
    """Liger Kernel replacement options."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = False
    stage: Literal["pre_build", "post_build"] = "pre_build"
    modules: Literal["auto"] | dict[str, bool] = "auto"
```

Use this YAML shape:

```yaml
model:
  liger_kernel:
    enabled: false
    stage: "pre_build"
    modules: "auto"
```

For explicit module selection:

```yaml
model:
  liger_kernel:
    enabled: true
    stage: "post_build"
    modules:
      rms_norm: true
      rope: true
      swiglu: true
      layer_norm: false
      cross_entropy: false
      fused_linear_cross_entropy: false
```

Treat `"auto"` as recipe-owned model-family resolution. Do not silently enable
loss kernels in `"auto"`.

### 2. Implement Recipe-Local Liger Helpers

Put Liger integration in:

```text
recipes/<recipe>/model/liger.py
```

Recommended public helpers:

```python
def apply_liger_kernel_pre_build(*, model_name_or_path: str, config: LigerKernelConfig) -> None:
    """Apply official Liger model-family monkey patches before model construction."""


def patch_liger_kernel_post_build(model: torch.nn.Module, *, config: LigerKernelConfig) -> torch.nn.Module:
    """Replace supported modules on an already-built model instance."""
```

Use the official pre-build model-family API when available. This changes
Transformers classes/functions in the current Python process before
`from_pretrained(...)`; it does not edit source files on disk.

Use post-build replacement only in recipe-local code with explicit supported
module types or paths. Do not add a generic repository-wide `named_modules()`
rewriter that replaces unknown modules by class-name guesses.

### 3. Wire Prepare Model

Place pre-build Liger before model construction:

```python
liger_config = self.config.model.liger_kernel
if liger_config.enabled and liger_config.stage == "pre_build":
    apply_liger_kernel_pre_build(
        model_name_or_path=self.config.model.pretrained_model_name_or_path,
        config=liger_config,
    )

model = self.model_kit.build_model(...).to(self.device)
```

Place post-build Liger in the recipe model patch list:

```python
model_patches = [patch_qwen3vl_conv3d, patch_qwen3vl_model_flops]
if liger_config.enabled and liger_config.stage == "post_build":
    model_patches.append(partial(patch_liger_kernel_post_build, config=liger_config))

model = self.model_kit.apply_model_patches(model, model_patches)
```

Keep ordering:

```text
pre-build Liger -> load model -> recipe/post-build patches -> token-loss patch
-> freeze policy -> trainable dtype upcast -> checkpointing -> compile
-> parallelize -> build optimizer
```

### 4. Resolve Modules Conservatively

`modules="auto"` should resolve from the recipe's known model family. If the
family is unsupported, fail with a clear error.

Supported module names should be semantic, not raw class names:

```text
rms_norm
layer_norm
rope
swiglu
geglu
cross_entropy
fused_linear_cross_entropy
```

For post-build replacements:

- preserve parameter names and checkpoint keys;
- copy or reuse existing weights, dtype, device, and `requires_grad`;
- replace only modules that the recipe explicitly supports;
- record applied replacements on the model, such as
  `model._mvp_engine_liger_kernel = {...}`;
- fail when an explicitly requested module cannot be applied.

### 5. Protect Token-Loss Semantics

Do not enable `cross_entropy` or `fused_linear_cross_entropy` when the recipe
also applies `TokenNormedLossKit.apply_chunked_token_loss_patch(...)`, unless
the recipe adds a dedicated compatibility path that still returns unreduced
per-token loss.

Default behavior:

- `modules="auto"` keeps Liger loss kernels disabled;
- explicit loss-kernel requests fail when token-normalized loss is active and no
  compatibility code exists;
- non-loss kernels may be used before the token-loss patch.

## Validation

### Soft Validation

- config exposes `model.liger_kernel.enabled`, `stage`, and `modules`;
- no Liger dependency is added to `pyproject.toml` unless the user requested it;
- pre-build patching runs before `build_model(...)`;
- post-build patching runs through `MLLMModelKit.apply_model_patches(...)`;
- post-build replacements are recipe-local and explicit;
- `"auto"` module resolution is tied to model family and fails on unsupported
  families;
- loss kernels cannot silently conflict with token-normalized loss;
- Liger runs before freeze, checkpointing, compile, and distributed wrapping.

### Hard Validation

Copy `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/liger-kernel/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run structure and smoke tests. For runtime validation, run at least one smoke
with Liger enabled for the configured stage:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q --config-override model.liger_kernel.enabled=true
```

If both stages are implemented, validate both:

```bash
pytest recipes/<recipe>/tests/test_smoke.py -q \
  --config-override model.liger_kernel.enabled=true \
  --config-override model.liger_kernel.stage=pre_build
pytest recipes/<recipe>/tests/test_smoke.py -q \
  --config-override model.liger_kernel.enabled=true \
  --config-override model.liger_kernel.stage=post_build
```

If Liger Kernel is not installed or the environment lacks required GPU/NPU
support, report the exact command that should be run in the real environment.

## Output

- State target recipe and configured stage.
- State resolved modules and whether they came from `"auto"` or explicit config.
- State whether official pre-build API or recipe-local post-build replacement
  was used.
- State loss-kernel compatibility with token-normalized loss.
- Report validation commands and remaining runtime gaps.

## Read On Demand

- `skills/kit/mllm-model-kit/SKILL.md`: model patch placement and ordering.
- `skills/kit/token-loss-kit/SKILL.md`: token-loss patch contract.
- `references/asserts.py`: recipe-local structure assertions.
