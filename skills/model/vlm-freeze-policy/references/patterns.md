# VLM Freeze Policy Patterns

Use this reference when defining parameter groups or deciding build order.

## Parameter Groups

Start from the real model:

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

Common group examples:

- vision: `visual`, `vision_tower`, `vision_model`, `vit`, `image_encoder`;
- connector: `merger`, `projector`, `multi_modal_projector`, `resampler`,
  `q_former`, `adapter`;
- language: `language_model`, `model.layers`, `text_model`, `lm_head`,
  `embed_tokens`.

Use exact prefixes when possible:

```python
VISION_PREFIXES = ("model.visual.",)
CONNECTOR_PREFIXES = ("model.visual.merger.",)
LANGUAGE_PREFIXES = ("model.language_model.", "lm_head.")
```

Keep connector prefixes more specific than vision prefixes if one path is nested
inside another. Check overlap with `named_parameters()` before trusting the
policy.

## Freeze Helper

Keep the helper local to the recipe model module:

```python
def apply_freeze_policy(model, *, freeze_vision: bool, freeze_connector: bool, freeze_language: bool):
    for name, parameter in model.named_parameters():
        if freeze_connector and name.startswith(CONNECTOR_PREFIXES):
            parameter.requires_grad = False
        elif freeze_vision and name.startswith(VISION_PREFIXES):
            parameter.requires_grad = False
        elif freeze_language and name.startswith(LANGUAGE_PREFIXES):
            parameter.requires_grad = False
```

Order matters when prefixes overlap. Match the more specific group first.

## Build Order

Recommended order:

1. load model;
2. apply compatibility patches, forward injections, and checkpointing hooks;
3. apply freeze policy;
4. upcast or count trainable parameters;
5. compile if enabled;
6. parallelize model;
7. build optimizer from parameters where `requires_grad` is true.

## FLOPs And MFU

Freeze-aware FLOPs depend on gradient flow:

- trainable module: forward + input-gradient + weight-gradient;
- frozen module with gradients passing through inputs: forward + input-gradient;
- frozen module with no gradient path through it: forward only.

Use component freeze flags where FLOPs are computed. Do not patch MFU logging
with freeze assumptions after FLOPs have already been estimated.
