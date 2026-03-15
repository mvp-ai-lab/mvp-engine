---
name: model-flops-utilization
description: Add or correct `calculate_model_flops(...)` for the current model and validate that the method is callable, returns the expected contract, and generalizes across Transformer and ViT-style architectures. Use when a model needs MFU support implemented, not just documented.
---

# Model FLOPs Utilization

Implement `calculate_model_flops(...)` on the target model. Do not stop at formulas or pseudo-code.
**中文：** [SKILL.md](../../../zh-cn/model/model-flops-utilization/SKILL.md)

## Goal

- Add `calculate_model_flops(...) -> float` to the current model or its local adapter class.
- Use explicit shape inputs instead of hidden runtime state.
- Validate that the method is callable, returns a `float`, and handles boundary cases.
- Leave behind example references that show the pattern on both Transformer and ViT models.

## 1. Classify the target model before editing

- Prefer the in-place path when the model class lives in this repo and is owned by the current recipe.
- Use the local-adapter path when the runtime model class comes from a third-party package such as `transformers`.
- Use the architecture-specific template that matches the real repeated compute structure.

### In-place path

Use this path when all of the following are true:

- The model class is defined in the current repo.
- The recipe already instantiates that class directly.
- Adding one method does not force unrelated refactors.

### Local-adapter path

Use this path when any of the following are true:

- The model class comes from a third-party package.
- Editing vendor code is not acceptable.
- The recipe already has a thin wrapper or can safely swap to a local subclass.

For the local-adapter path, define a local subclass with `calculate_model_flops(...)` and replace the instantiation site to use that subclass.

## 2. Implement the method, not just the formula

Every successful use of this skill must produce a runnable method with this contract:

```python
def calculate_model_flops(
    self,
    *,
    batch_size: int,
    seq_len: int | None = None,
    image_size: int | tuple[int, int] | None = None,
    patch_size: int | tuple[int, int] | None = None,
    is_training: bool = True,
) -> float:
    ...
```

Rules:

- Transformer: require `batch_size`, `seq_len`, and `is_training`.
- ViT: require `batch_size`, `image_size`, `patch_size`, and `is_training`.
- Raise `ValueError` for missing required shape inputs.
- Return one `float`: FLOPs per process per step.
- If you compute a breakdown dict for debugging, keep it local; do not return it instead of the `float`.

## 3. Use the matching implementation template

### Transformer template

Use this for dense encoder, decoder-only, or encoder-decoder stacks when FLOPs are dominated by attention and MLP blocks.

```python
def calculate_model_flops(
    self,
    *,
    batch_size: int,
    seq_len: int | None = None,
    image_size: int | tuple[int, int] | None = None,
    patch_size: int | tuple[int, int] | None = None,
    is_training: bool = True,
) -> float:
    if seq_len is None:
        raise ValueError("Transformer FLOPs requires seq_len.")

    B = int(batch_size)
    S = int(seq_len)
    L = int(self.config.num_hidden_layers)
    H = int(self.config.hidden_size)
    I = int(self.config.intermediate_size)

    per_layer = 8 * B * S * H * H + 4 * B * S * S * H + 4 * B * S * H * I
    transformer_flops = L * per_layer

    lm_head_flops = 0.0
    if hasattr(self.config, "vocab_size"):
        V = int(self.config.vocab_size)
        lm_head_flops = 2 * B * S * H * V

    forward_flops = float(transformer_flops + lm_head_flops)
    return forward_flops * 3.0 if is_training else forward_flops
```

### ViT template

Use this for patch-embedding vision transformers and keep the patch/head assumptions explicit.

```python
def calculate_model_flops(
    self,
    *,
    batch_size: int,
    seq_len: int | None = None,
    image_size: int | tuple[int, int] | None = None,
    patch_size: int | tuple[int, int] | None = None,
    is_training: bool = True,
) -> float:
    if image_size is None or patch_size is None:
        raise ValueError("ViT FLOPs requires image_size and patch_size.")

    B = int(batch_size)
    if isinstance(image_size, int):
        img_h, img_w = image_size, image_size
    else:
        img_h, img_w = map(int, image_size)
    if isinstance(patch_size, int):
        p_h, p_w = patch_size, patch_size
    else:
        p_h, p_w = map(int, patch_size)

    if min(B, img_h, img_w, p_h, p_w) <= 0:
        raise ValueError("batch_size, image_size, and patch_size must be > 0")
    if img_h % p_h != 0 or img_w % p_w != 0:
        raise ValueError("image_size must be divisible by patch_size")

    N = (img_h // p_h) * (img_w // p_w)
    C = int(getattr(self.config, "num_channels", 3))
    D = int(self.config.hidden_size)
    L = int(self.config.num_hidden_layers)
    I = int(self.config.intermediate_size)
    K = int(getattr(self.config, "num_labels", 1000))

    patch_embed_flops = 2 * B * N * (C * p_h * p_w) * D
    block_flops = 8 * B * N * D * D + 4 * B * N * N * D + 4 * B * N * D * I
    backbone_flops = L * block_flops
    head_flops = 2 * B * D * K

    forward_flops = float(patch_embed_flops + backbone_flops + head_flops)
    return forward_flops * 3.0 if is_training else forward_flops
```

### Local-adapter template for external models

```python
class ExternalModelWithFlops(ExternalModel):
    def calculate_model_flops(... ) -> float:
        ...

# Replace:
# model = ExternalModel(config)
# With:
# model = ExternalModelWithFlops(config)
```

## 4. Validate immediately after implementation

Do not claim success until the method is exercised.

Validation checklist:

1. The target model instance actually exposes `calculate_model_flops`.
2. The signature includes the architecture-required explicit parameters.
3. `is_training=True` and `is_training=False` both run.
4. Return type is `float`.
5. FLOPs are positive.
6. Training FLOPs are greater than or equal to eval FLOPs.
7. Missing required shape inputs raise `ValueError` or `TypeError`.

## 5. Archive examples instead of leaving validation scattered

Keep example implementations and tests under `references/` so the skill remains reusable.

Reference examples in this skill:

- `references/external_vit/`: local subclass pattern for a third-party ViT.
- `references/decoder_transformer/`: direct method pattern for a decoder-style transformer.
- `references/validation_cases.py`: prompt set for trigger and contract validation.
- `references/run_validation.py`: dry-run template generator.
- `references/check_acceptance.py`: acceptance gate checker.

## Pitfalls

- Do not answer with formulas only when the task asks for implementation.
- Do not return dict-only output; the primary contract is a single `float`.
- Do not infer required shape inputs from hidden runtime tensors when the skill contract requires explicit parameters.
- Do not patch third-party package source directly when a local subclass is sufficient.
- Do not skip execution checks; a method that exists but does not run is still a failure.
- Do not assume one formula covers MoE, sparse attention, fused kernels, or activation recomputation; note those as exclusions.

## Reference

- External ViT example: `references/external_vit/`
- Decoder transformer example: `references/decoder_transformer/`
- Validation prompts: `references/validation_cases.py`
