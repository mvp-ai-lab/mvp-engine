# TP Module Config Playbook (EN)

## Goal
Generate `<MODEL_NAME>_TP_MODULE_CONFIG` for a new model under `recipes/`, then bind `TP_MODULE_CONFIG = <MODEL_NAME>_TP_MODULE_CONFIG` on the model class.

## Runtime Contract in This Repo
- Runtime entry: `mvp_engine/distributed/tp.py`.
- Required format: `dict[str, object]` mapping `module.__class__.__name__ -> plan`.
- Plan format: `dict[child_linear_name, "col" | "row"]`.
- Child names must match `named_children()` on the target class.

## Analysis Procedure
1. Open target `modeling_*.py` in `recipes/**/model/**/`.
2. Find the top-level model class used by training.
3. Find compute block classes instantiated repeatedly (attention, MLP, branch MLP, projector).
4. In each block class, collect direct `nn.Linear` child names from `__init__`.
5. Assign TP mode with these heuristics:
- Use `"col"` for input-expansion projections: `q_proj`, `k_proj`, `v_proj`, `qkv`, `fc1`, `up_proj`, `gate_proj`, and branch variants like `_a/_b`.
- Use `"row"` for output-merge projections: `out_proj`, `o_proj`, `proj_out`, `fc2`, `down_proj`, `wo`, and branch variants like `_a/_b`.
- If unsure, treat early projections as `"col"` and final projection back to hidden size as `"row"`.

## Editing Template
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

## Validation Checklist
- Config class keys equal real runtime class names (`module.__class__.__name__`).
- Each plan key exists in the class as a child module.
- Plan values only use `"col"` or `"row"`.
- `TP_MODULE_CONFIG` is defined on the top-level model class.

## TomatoViT Reference
Use `recipes/tomatovit/model/tomato_vit/modeling_tomatovit.py` as a canonical pattern:
- Constant: `TOMATOVIT_TP_MODULE_CONFIG`
- Binding: `TP_MODULE_CONFIG = TOMATOVIT_TP_MODULE_CONFIG`
