---
name: tensor-parallel
description: Add recipe-local tensor parallel plans and optional TP postprocess hooks for a
  model in this repo. Use when enabling TP for a new model, updating mesh config, or fixing
  TP-local runtime metadata.
---

# TP Module Config Playbook

## Goal

- Generate `<MODEL_NAME>_TP_MODULE_CONFIG` for the target model and bind it on the
  top-level model class as `TP_MODULE_CONFIG`.
- Add `TP_MODULE_POSTPROCESSORS` only when TP sharding changes module-local metadata that
  the runtime does not fix automatically.
- Update the training mesh config so TP size, replicate, and shard are compatible.

## Required Inputs

- The target `modeling_*.py` file under `recipes/**/model/**/`.
- The top-level model class actually used by training.
- The repeated compute block classes that contain the linears TP should shard.
- The current training config and mesh settings. If config changes are needed,
  include GPUs per node and the TP size that should be set.

## Workflow

### 1. Collect the runtime structure

- Find the target modeling file and the top-level model class used by training.
- Find the repeated compute blocks such as attention, MLP, projector, or branch MLP
  classes.
- In each block class, collect direct `nn.Linear` child names from `__init__`.
- Build the TP plan with these heuristics:
  - use `"col"` for input-expansion projections such as `q_proj`, `k_proj`, `v_proj`,
    `qkv`, `fc1`, `up_proj`, `gate_proj`, and `_a/_b` branch variants
  - use `"row"` for output-merge projections such as `out_proj`, `o_proj`, `proj_out`,
    `fc2`, `down_proj`, `wo`, and `_a/_b` branch variants
  - if unsure, treat early projections as `"col"` and the final projection back to hidden
    size as `"row"`
- Keep in mind the runtime contract in this repo:
  - `TP_MODULE_CONFIG` maps `module.__class__.__name__ -> plan`
  - each plan maps child linear names to `"col"` or `"row"`
  - child names must match `named_children()` on the real module class

### 2. Implement the modeling-side TP config

- Define `<MODEL_NAME>_TP_MODULE_CONFIG` in the modeling file.
- Bind it on the top-level model class as `TP_MODULE_CONFIG`.
- If the model comes from `transformers`, it is acceptable to create a wrapper class with
  the same top-level class name in the local modeling file and bind the TP attributes
  there.
- If the modeling file already contains the top-level wrapper class used by training, only
  extend that existing class with `TP_MODULE_CONFIG` or `TP_MODULE_POSTPROCESSORS`; do not
  create a second wrapper class with the same name.
- If the model needs both TP and FSDP2 prefetching, `TP_MODULE_CONFIG`,
  `TP_MODULE_POSTPROCESSORS`, and `APPLY_FSDP2_CUSTOM_PREFETCHING` must be merged onto the
  same top-level model class declaration.

```python
<MODEL_NAME>_TP_MODULE_CONFIG: dict[str, object] = {
    "<AttentionClass>": {
        "q_proj": "col",
        "k_proj": "col",
        "v_proj": "col",
        "out_proj": "row",
    },
    "<MLPClass>": {
        "fc1": "col",
        "fc2": "row",
    },
}


class <TopModelClass>(...):
    TP_MODULE_CONFIG = <MODEL_NAME>_TP_MODULE_CONFIG
```

### 3. Check whether TP postprocessing is required

- Read the target module's `forward()` carefully after drafting the TP plan.
- If `forward()` only consumes tensor shapes produced by the sharded linears, extra
  postprocessing is usually unnecessary.
- If `forward()` depends on cached metadata on `self`, add a postprocess hook.
- Common warning signs include:
  - `view(..., self.num_attention_heads, self.attention_head_size)`
  - `reshape(..., self.num_key_value_heads, ...)`
  - `split(self.hidden_size, dim=...)`
  - loops or indexing that assume global expert, head, or group counts

### 4. Add TP postprocessing when needed

- Add a recipe-local helper and bind it through `TP_MODULE_POSTPROCESSORS`.
- The dict key must match the runtime class name, just like `TP_MODULE_CONFIG` keys do.
- Keep the hook minimal: update only the fields whose meaning changes after sharding.
- Prefer changing module-local derived metadata instead of mutating model config.

```python
def _adjust_attention_for_tp(module, tp_mesh) -> None:
    tp_size = tp_mesh.size()
    if tp_size <= 1:
        return
    module.num_attention_heads //= tp_size
    module.all_head_size = module.num_attention_heads * module.attention_head_size


class MyModel(...):
    TP_MODULE_CONFIG = MYMODEL_TP_MODULE_CONFIG
    TP_MODULE_POSTPROCESSORS = {
        "MyAttention": _adjust_attention_for_tp,
    }
```

### 5. Update the training config

- If the user has not already specified them, ask these two questions before editing
  config:
  - how many GPUs per node will training use
  - what TP size should the recipe use
- Add `tensor: <N>` to the mesh config when it is missing.
- Adjust `replicate` and `shard` so they remain compatible with the chosen TP size.

The final structure should look like:

```yaml
parallel:
  mesh:
    replicate: <D>
    shard: <S>
    tensor: <N>
  backend_kwargs:
    ...
```

## Validation

- `TP_MODULE_CONFIG` keys equal real runtime class names.
- Each plan key exists in the target class as a real child module.
- Plan values use only `"col"` or `"row"`.
- The top-level model class exposes `<MODEL_NAME>_TP_MODULE_CONFIG` through
  `TP_MODULE_CONFIG`.
- If the top-level wrapper class already existed, the change extends that class instead of
  creating a second class with the same name.
- If the model uses both TP and FSDP2 prefetching, the related class attributes are merged
  onto the same top-level model class declaration.
- Every module whose `forward()` depends on cached global metadata was reviewed for TP
  postprocessing.
- `TP_MODULE_POSTPROCESSORS`, if present, uses real runtime class names and only mutates
  local runtime metadata.
- The mesh config has compatible `replicate`, `shard`, and `tensor` values.

Add recipe-local tests under `recipes/<recipe>/skill_tests/tensor-parallel/`:

- `test_structure.py`: verify recipe structure and core wiring.
- `test_smoke.py`: run one real recipe-owned training step and checkpoint/log path.
- `test_effectiveness.py`: create a recipe-local test that uses
  `mvp_engine.test.recipe_probe` helpers, then add a method such as
  `assert_tp_tensor_dims_match_mesh(model, reference_shapes, tp_config, mesh)`.
  Compare each TP-covered parameter's local shape against its pre-parallel
  reference shape. Use mesh `tensor` size as `tp_size`. For `"col"`, check the
  col-sharded dim with denominator `tp_size`. For `"row"`, check the row-sharded
  dim with denominator `tp_size * fsdp_shard_size` when FSDP2
  sharding is also enabled. Compare DTensor parameters with `param.to_local().shape`; otherwise use `param.shape`.
  When every TP-plan-covered parameter's local shape matches the expected shape,
  the effectiveness test can be treated as passing.

## Output

- State which modeling and config files were updated.
- Summarize the TP plan by module class.
- State whether TP postprocessing was added and for which runtime classes.
- State the final mesh settings or the remaining user input needed to finish them.

## Read On Demand

- Read `./references/vit_classification/` when you need a full TP example with model
  changes, config wiring, and recipe-local tests.
