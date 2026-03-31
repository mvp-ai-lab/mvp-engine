# TP Module Config 操作手册（中文）

## 目标
为 `recipes/` 下的新模型生成 `<MODEL_NAME>_TP_MODULE_CONFIG`，然后在模型类上绑定 `TP_MODULE_CONFIG = <MODEL_NAME>_TP_MODULE_CONFIG`。

## 本仓库中的运行时约定
- 运行时入口：`mvp_engine/distributed/tp.py`。
- 必需格式：`dict[str, object]`，用于映射 `module.__class__.__name__ -> plan`。
- Plan 格式：`dict[child_linear_name, "col" | "row"]`。
- 子模块名必须与目标类上的 `named_children()` 一致。
- 可选的后处理格式：`dict[str, callable]`，绑定到顶层模型类的 `TP_MODULE_POSTPROCESSORS`。
- 后处理 callable 会在 `parallelize_module(module, tp_mesh, plan)` 之后调用，用于修正 TP 不会自动改写的模块局部元数据。

## 步骤

### 1. 收集信息
- 在 `recipes/**/model/**/` 下找到目标 `modeling_*.py`。
- 找到训练实际使用的顶层模型类。
- 找出会被重复实例化的计算块类（attention、MLP、branch MLP、projector）。
- 在每个 block 类中，收集 `__init__` 里直接定义的 `nn.Linear` 子模块名。
- 按以下启发式规则分配 TP 模式：
    - 输入扩张投影使用 `"col"`：`q_proj`、`k_proj`、`v_proj`、`qkv`、`fc1`、`up_proj`、`gate_proj`，以及带 `_a/_b` 的分支变体。
    - 输出聚合投影使用 `"row"`：`out_proj`、`o_proj`、`proj_out`、`fc2`、`down_proj`、`wo`，以及带 `_a/_b` 的分支变体。
    - 如果拿不准，前面的投影通常按 `"col"` 处理，最终回到 hidden size 的投影按 `"row"` 处理。

### 2. 修改 modeling 代码
- 在 modeling 文件中实现 `<MODEL_NAME>_TP_MODULE_CONFIG`。模板如下：
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
- 如果模型来自 `transformers` 的现有实现，可以在 modeling 文件中创建一个与原始模型类同名的 wrapper class，并在该类上绑定 `TP_MODULE_CONFIG`。
- 如果 modeling 文件里已经存在训练实际使用的顶层 wrapper class，只能在这个已有类上追加 `TP_MODULE_CONFIG` 或 `TP_MODULE_POSTPROCESSORS`，禁止再创建第二个同名 wrapper class。
- 如果模型同时需要 TP 与 FSDP2 prefetching，必须把 `TP_MODULE_CONFIG`、`TP_MODULE_POSTPROCESSORS` 与 `APPLY_FSDP2_CUSTOM_PREFETCHING` 合并到同一个顶层模型类声明中。

### 2.1 检查是否需要 TP 后处理
- 初步写完 TP plan 后，仔细阅读目标模块的 `forward()`。
- 如果 `forward()` 只消费并行化后的 linear 输出张量形状，通常不需要额外后处理。
- 如果 `forward()` 依赖缓存于 `self` 上的元数据，则大概率需要 TP 后处理 hook。
- 常见需要按本地 shard 调整的字段：
    - Attention 元数据：`num_attention_heads`、`num_key_value_heads`、`num_key_value_groups`、`all_head_size`
    - 分片尺寸：`hidden_size_per_partition`、`inner_dim`、基于 `head_dim` 派生并缓存的值
    - split/reshape 元数据：预计算 chunk size、切片边界、grouped projection 数量
    - 假设全局 head 数或全局 hidden 宽度的 cache/rope helper
- `forward()` 中的强提示信号：
    - `view(..., self.num_attention_heads, self.attention_head_size)`
    - `reshape(..., self.num_key_value_heads, ...)`
    - `split(self.hidden_size, dim=...)`
    - 基于缓存的 expert/head/group 数量做循环或索引

### 2.2 需要时添加 TP 后处理
- 如果模块需要修正运行时元数据，添加 recipe-local helper，并通过 `TP_MODULE_POSTPROCESSORS` 绑定。
- key 必须与运行时类名一致，和 `TP_MODULE_CONFIG` 的规则相同。
- hook 要尽量最小化：只更新那些在分片后语义发生变化的字段。
- 优先修改模块局部的派生字段，不要修改 model config。
- 示例：
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
- 本仓库中的 ViT 就需要这种修正，因为 `ViTSelfAttention.forward()` 会在 `q/k/v` 被切分后，继续使用缓存的 head 元数据做 reshape。

### 3. 修改训练配置
- 修改配置前，如果用户之前没有明确说明，你必须先问两个问题：
    - 单机训练会使用多少张 GPU？
    - 训练时 TP size 计划设为多少？（通常建议小于单机 GPU 数）
- 如果 mesh 配置里还没有 `tensor: <N>`，补上它。
- 同时修正 `replicate` 和 `shard`，让它们与新的 TP size 兼容。
- 最终配置结构应类似：
    ```yaml
    parallel:
      mesh:
        replicate: <D>
        shard: <S>
        tensor: <N>
      backend_kwargs:
        ...
    ```

## 验证清单
- [ ] 配置中的类名 key 与真实运行时类名一致（`module.__class__.__name__`）。
- [ ] 每个 plan key 都是该类中的真实子模块名。
- [ ] Plan value 只使用 `"col"` 或 `"row"`。
- [ ] `<MODEL_NAME>_TP_MODULE_CONFIG` 已绑定到顶层模型类。
- [ ] 若顶层 wrapper class 已存在，本次修改是在原类上追加属性，而不是新建第二个同名类。
- [ ] 若模型同时启用 TP 与 FSDP2 prefetching，相关类属性已合并到同一个顶层模型类声明中。
- [ ] 所有在 `forward()` 中使用缓存全局元数据的模块都已检查是否需要 TP 后处理。
- [ ] 如果存在 `TP_MODULE_POSTPROCESSORS`，其中的 key 必须与真实运行时类名一致。
- [ ] 后处理 hook 只修改本地运行时元数据，不得改写预训练参数张量。

## 示例
- 一个完整的 ViT TP 示例归档在 `./references/vit_classification/`，其中包含启用 TP 的模型文件、训练配置和 recipe-local 测试。
