# TP Module Config Playbook (EN)

## Goal
Generate `<MODEL_NAME>_TP_MODULE_CONFIG` for a new model under `recipes/`, then bind `TP_MODULE_CONFIG = <MODEL_NAME>_TP_MODULE_CONFIG` on the model class.

## Runtime Contract in This Repo
- Runtime entry: `mvp_engine/distributed/tp.py`.
- Required format: `dict[str, object]` mapping `module.__class__.__name__ -> plan`.
- Plan format: `dict[child_linear_name, "col" | "row"]`.
- Child names must match `named_children()` on the target class.
- Optional postprocess format: `dict[str, callable]` bound as `TP_MODULE_POSTPROCESSORS` on the top-level model class.
- A postprocess callable is invoked after `parallelize_module(module, tp_mesh, plan)` and should fix module-local metadata that TP does not rewrite automatically.

## Steps

### 1. Collecting Data
- Find target `modeling_*.py` in `recipes/**/model/**/`.
- Find the top-level model class used by training.
- Find compute block classes instantiated repeatedly (attention, MLP, branch MLP, projector).
- In each block class, collect direct `nn.Linear` child names from `__init__`.
- Assign TP mode with these heuristics:
    - Use `"col"` for input-expansion projections: `q_proj`, `k_proj`, `v_proj`, `qkv`, `fc1`, `up_proj`, `gate_proj`, and branch variants like `_a/_b`.
    - Use `"row"` for output-merge projections: `out_proj`, `o_proj`, `proj_out`, `fc2`, `down_proj`, `wo`, and branch variants like `_a/_b`.
    - If unsure, treat early projections as `"col"` and final projection back to hidden size as `"row"`.

### 2. Edit the Modeling Code
- Implement `<MODEL_NAME>_TP_MODULE_CONFIG` in the modeling file. Template:
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
- If the model is an exising model in `transformers`, you can create a wrapper class with the same name as the original model's class in the modeling file and bind `TP_MODULE_CONFIG` there.
- If the modeling file already contains the top-level wrapper class used by training, only extend that existing class with `TP_MODULE_CONFIG` or `TP_MODULE_POSTPROCESSORS`; do not create a second wrapper class with the same name.
- If the model needs both TP and FSDP2 prefetching, `TP_MODULE_CONFIG`, `TP_MODULE_POSTPROCESSORS`, and `APPLY_FSDP2_CUSTOM_PREFETCHING` must be merged onto the same top-level model class declaration.

### 2.1 Check Whether TP Postprocessing Is Required
- After drafting the TP plan, read the target module's `forward()` carefully.
- If `forward()` only consumes tensor shapes produced by the parallelized linears, no extra postprocess is usually needed.
- If `forward()` depends on metadata cached on `self`, you likely need a TP postprocess hook.
- Common fields that need local-shard adjustment:
    - Attention metadata: `num_attention_heads`, `num_key_value_heads`, `num_key_value_groups`, `all_head_size`
    - Partition sizes: `hidden_size_per_partition`, `inner_dim`, `head_dim`-derived cached values
    - Split/reshape metadata: precomputed chunk sizes, slice boundaries, grouped projection counts
    - Cache/rope helpers that assume global head counts or global hidden widths
- Strong warning signs in `forward()`:
    - `view(..., self.num_attention_heads, self.attention_head_size)`
    - `reshape(..., self.num_key_value_heads, ...)`
    - `split(self.hidden_size, dim=...)`
    - loops or indexing based on cached expert/head/group counts

### 2.2 Add TP Postprocessing When Needed
- If a module needs runtime metadata fixes, add a recipe-local helper and bind it through `TP_MODULE_POSTPROCESSORS`.
- The key must match the runtime class name, same as `TP_MODULE_CONFIG`.
- Keep the hook minimal: only update fields whose meaning changes after sharding.
- Prefer updating module-local derived fields instead of mutating the model config.
- Example:
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
- ViT in this repo needs exactly this kind of fix because `ViTSelfAttention.forward()` reshapes with cached head metadata after `q/k/v` are sharded.

### 3. Edit the Training Config
- Before editing the config, you MUST ask the user two questions if they are not mentioned by the user before:
    - How many GPUs per node will be used for training?
    - How many TP size should be used for training? (It is recommended to be smaller than the number of GPUs per node)
- If the mesh config doesn't already have `tensor: <N>`, add it.
- Fix the `replicate` and `shard` values to be compatible with the new TP size.
- The final config structure should look like this:
    ```yaml
    parallel:
      mesh:
        replicate: <D>
        shard: <S>
        tensor: <N>
      backend_kwargs:
        ...
    ```

## Validation Checklist
- [ ] Config class keys equal real runtime class names (`module.__class__.__name__`).
- [ ] Each plan key exists in the class as a child module.
- [ ] Plan values only use `"col"` or `"row"`.
- [ ] `<MODEL_NAME>_TP_MODULE_CONFIG` is defined on the top-level model class.
- [ ] If the top-level wrapper class already existed, this change extended that class instead of creating a second class with the same name.
- [ ] If the model uses both TP and FSDP2 prefetching, the related class attributes are merged onto the same top-level model class declaration.
- [ ] Every module whose `forward()` uses cached global metadata has been reviewed for TP postprocessing.
- [ ] `TP_MODULE_POSTPROCESSORS` keys, if present, equal real runtime class names.
- [ ] Postprocess hooks only change local runtime metadata and do not mutate pretrained parameter tensors.

## Example
- A full ViT TP example is archived under `./references/vit_classification/`, including the TP-enabled model file, training config, and recipe-local tests.
