---
name: tensor-parallel
description: 为本仓库中的模型增加 recipe-local 的 tensor parallel plan 和可选 TP 后处理 hook。适用于给新模型接入 TP、更新 mesh 配置，或修正 TP 后的模块局部元数据。
---

# TP Module Config 操作手册

## Goal

- 为目标模型生成 `<MODEL_NAME>_TP_MODULE_CONFIG`，并在顶层模型类上绑定为 `TP_MODULE_CONFIG`。
- 只有当 TP 分片改变了运行时不会自动修正的模块局部元数据时，才增加 `TP_MODULE_POSTPROCESSORS`。
- 调整训练 mesh 配置，使 TP size、replicate 和 shard 彼此兼容。

## Required Inputs

- `recipes/**/model/**/` 下的目标 `modeling_*.py` 文件。
- 训练实际使用的顶层模型类。
- 需要做 TP 分片的重复计算 block 类。
- 当前训练配置和 mesh 设置。
- 如果需要修改配置且用户尚未明确说明：
  - 单机 GPU 数量
  - 目标 TP size

## Workflow

### 1. 收集运行时结构

- 找到目标 modeling 文件和训练实际使用的顶层模型类。
- 找到会重复实例化的计算块，例如 attention、MLP、projector 或 branch MLP。
- 在每个 block 类中，收集 `__init__` 里直接定义的 `nn.Linear` 子模块名。
- 按以下启发式规则建立 TP plan：
  - `q_proj`、`k_proj`、`v_proj`、`qkv`、`fc1`、`up_proj`、`gate_proj` 以及 `_a/_b` 分支变体通常用 `"col"`
  - `out_proj`、`o_proj`、`proj_out`、`fc2`、`down_proj`、`wo` 以及 `_a/_b` 分支变体通常用 `"row"`
  - 如果拿不准，前面的扩张投影通常按 `"col"`，回到 hidden size 的最终投影按 `"row"`
- 同时遵守本仓库的运行时约定：
  - `TP_MODULE_CONFIG` 映射 `module.__class__.__name__ -> plan`
  - 每个 plan 把线性层子模块名映射到 `"col"` 或 `"row"`
  - 子模块名必须与真实类上的 `named_children()` 一致

### 2. 实现 modeling 侧 TP 配置

- 在 modeling 文件里定义 `<MODEL_NAME>_TP_MODULE_CONFIG`。
- 在顶层模型类上绑定 `TP_MODULE_CONFIG`。
- 如果模型来自 `transformers`，可以在本地 modeling 文件里创建同名 wrapper class，并在其上绑定 TP 属性。

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

### 3. 检查是否需要 TP 后处理

- 在写完初版 TP plan 后，仔细阅读目标模块的 `forward()`。
- 如果 `forward()` 只是消费分片后的线性层输出张量形状，通常不需要额外后处理。
- 如果 `forward()` 依赖缓存于 `self` 上的元数据，就应考虑增加 postprocess hook。
- 常见信号包括：
  - `view(..., self.num_attention_heads, self.attention_head_size)`
  - `reshape(..., self.num_key_value_heads, ...)`
  - `split(self.hidden_size, dim=...)`
  - 基于全局 expert、head 或 group 数量做循环或索引

### 4. 需要时添加 TP 后处理

- 添加 recipe-local helper，并通过 `TP_MODULE_POSTPROCESSORS` 绑定。
- 字典 key 必须和运行时类名一致，与 `TP_MODULE_CONFIG` 的规则相同。
- hook 要尽量小，只更新那些在分片后语义发生变化的字段。
- 优先修改模块局部派生元数据，不要改 model config。

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

### 5. 修改训练配置

- 如果用户还没有明确说明，在改配置前先问两个问题：
  - 单机训练会用多少张 GPU
  - 训练计划使用多大的 TP size
- 如果 mesh 配置缺少 `tensor: <N>`，补上它。
- 同时调整 `replicate` 和 `shard`，让它们与所选 TP size 兼容。

最终结构应类似：

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

- `TP_MODULE_CONFIG` 的 key 与真实运行时类名一致。
- 每个 plan key 都是目标类上的真实子模块。
- plan value 只使用 `"col"` 或 `"row"`。
- 顶层模型类通过 `TP_MODULE_CONFIG` 暴露了 `<MODEL_NAME>_TP_MODULE_CONFIG`。
- 所有依赖缓存全局元数据的模块都检查过是否需要 TP 后处理。
- 如果存在 `TP_MODULE_POSTPROCESSORS`，其 key 与真实运行时类名一致，且只修改本地运行时元数据。
- mesh 配置里的 `replicate`、`shard` 和 `tensor` 彼此兼容。

## Output

- 说明更新了哪些 modeling 和 config 文件。
- 按模块类总结 TP plan。
- 说明是否新增了 TP 后处理，以及作用于哪些运行时类。
- 说明最终 mesh 设置，或还缺哪些用户输入才能完成配置。

## Read On Demand

- 需要完整 TP 样例时，读取 `./references/vit_classification/`，其中包含模型改动、配置接线和 recipe-local 测试。
