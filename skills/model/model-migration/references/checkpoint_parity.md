# Checkpoint And Parity Patterns

Use this reference when validating migrated model behavior or checkpoint
compatibility.

## Source Identity

Record which source file was treated as authoritative:

```bash
sha256sum SOURCE_MODEL.py SOURCE_CONFIG.py
diff -u SOURCE_MODEL.py TARGET_MODEL.py
```

If the checkpoint directory contains its own modeling files, compare those too.
The checkpoint-adjacent source often reflects the exact implementation that
produced the weights.

## Strict State Dict Loading

Prefer strict loading:

```python
missing, unexpected = model.load_state_dict(state_dict, strict=False)
assert not missing
assert not unexpected
model.load_state_dict(state_dict, strict=True)
```

If strict loading fails, compare keys before changing checkpoints:

```python
source_keys = set(source_model.state_dict())
target_keys = set(target_model.state_dict())
assert source_keys == target_keys
```

Fix naming or module-structure drift in the migrated model before considering
checkpoint conversion.

## Behavior Parity

Use deterministic inputs and eval mode:

```python
source_model.eval()
target_model.eval()
with torch.no_grad():
    source_outputs = source_model(**inputs)
    target_outputs = target_model(**inputs)
torch.testing.assert_close(target_outputs.logits, source_outputs.logits)
```

For training-only layers, also compare one forward/backward step when feasible.
Keep tolerances explicit and tied to dtype.

## NPU Variant

For NPU-specific files:

- preserve class names or provide clearly documented aliases;
- keep state dict keys identical to the base implementation;
- guard `torch_npu` imports so CPU/GPU environments can still import the module;
- use fused NPU ops only when tensors are on NPU;
- keep a fallback path with the same math.
