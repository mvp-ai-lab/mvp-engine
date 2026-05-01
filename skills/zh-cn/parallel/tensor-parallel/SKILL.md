---
name: tensor-parallel
description: 为本仓库中的模型增加 recipe-local 的 tensor parallel plan 和可选 TP 后处理 hook。
  适用于给新模型接入 TP、更新 mesh 配置，或修正 TP 后的模块局部元数据。
---

# TP Module Config 操作手册

## Goal

- 为目标模型生成 `<MODEL_NAME>_TP_MODULE_CONFIG`，并在顶层模型类上绑定为
  `TP_MODULE_CONFIG`。
- 只有当 TP 分片改变了运行时不会自动修正的模块局部元数据时，才增加
  `TP_MODULE_POSTPROCESSORS`。
- 调整训练 mesh 配置，使 TP size、replicate 和 shard 彼此兼容。

## Required Inputs

- `recipes/**/model/**/` 下的目标 `modeling_*.py` 文件。
- 训练实际使用的顶层模型类。
- 需要做 TP 分片的重复计算 block 类。
- 当前训练配置和 mesh 设置；如果需要修改配置，应包含单机 GPU 数量以及需要设置的
  TP size。

## Workflow

### 1. 收集运行时结构

- 找到目标 modeling 文件和训练实际使用的顶层模型类。
- 找到会重复实例化的计算块，例如 attention、MLP、projector 或 branch MLP。
- 在每个 block 类中，收集 `__init__` 里直接定义的 `nn.Linear` 子模块名。
- 按以下启发式规则建立 TP plan：
  - `q_proj`、`k_proj`、`v_proj`、`qkv`、`fc1`、`up_proj`、`gate_proj` 以及 `_a/_b`
    分支变体通常用 `"col"`
  - `out_proj`、`o_proj`、`proj_out`、`fc2`、`down_proj`、`wo` 以及 `_a/_b`
    分支变体通常用 `"row"`
  - 如果拿不准，前面的扩张投影通常按 `"col"`，回到 hidden size 的最终投影按 `"row"`
- 同时遵守本仓库的运行时约定：
  - `TP_MODULE_CONFIG` 映射 `module.__class__.__name__ -> plan`
  - 每个 plan 把线性层子模块名映射到 `"col"` 或 `"row"`
  - 子模块名必须与真实类上的 `named_children()` 一致

### 2. 实现 modeling 侧 TP 配置

- 在 modeling 文件里定义 `<MODEL_NAME>_TP_MODULE_CONFIG`。
- 在顶层模型类上绑定 `TP_MODULE_CONFIG`。
- 如果模型来自 `transformers`，可以在本地 modeling 文件里创建同名 wrapper class，
  并在其上绑定 TP 属性。
- 如果 modeling 文件里已经存在训练实际使用的顶层 wrapper class，只能在这个已有类上追加
  `TP_MODULE_CONFIG` 或 `TP_MODULE_POSTPROCESSORS`，禁止再创建第二个同名 wrapper class。
- 如果模型同时需要 TP 与 FSDP2 prefetching，必须把 `TP_MODULE_CONFIG`、
  `TP_MODULE_POSTPROCESSORS` 与 `APPLY_FSDP2_CUSTOM_PREFETCHING` 合并到同一个顶层模型类声明中。

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
- 若顶层 wrapper class 已存在，本次修改是在原类上追加属性，而不是新建第二个同名类。
- 若模型同时启用 TP 与 FSDP2 prefetching，相关类属性已合并到同一个顶层模型类声明中。
- 所有依赖缓存全局元数据的模块都检查过是否需要 TP 后处理。
- 如果存在 `TP_MODULE_POSTPROCESSORS`，其 key 与真实运行时类名一致，且只修改本地运行时元数据。
- mesh 配置里的 `replicate`、`shard` 和 `tensor` 彼此兼容。

同时在 `recipes/<recipe>/skill_tests/tensor-parallel/` 下补 recipe-local 测试：

- `test_spec.yaml`：声明这个 skill 在该 recipe 上要求哪些测试层级，并包含
  `requires.effectiveness: true`。
- `test_structure.py`：至少验证 recipe import、registry 接线、config schema 校验、
  required slots，以及 logger/checkpoint hooks；还必须验证用户顶层模型类上的
  TP 类属性与配置接线存在。
- `test_runtime.py`：至少成功构建 dataset、collator、model、optimizer、
  scheduler 和 engine，且不直接启动训练；还必须验证运行时能解析
  `TP_MODULE_CONFIG` 并执行必需的 postprocessor。
- `test_smoke.py`：覆盖 1 个真实、recipe-owned 的 single step：forward、loss、
  backward、optimizer step、logger write，以及 checkpoint noop 或临时保存；
  还必须验证用户自己的 recipe / model 能在启用 TP 的情况下完成这一步。
- test_effectiveness.py: 参考 `tests/test_smoke_template.py` 创建 recipe-local
  `test_effectiveness.py`，增加一个新的方法，例如
  `assert_tp_tensor_dims_match_mesh(model, reference_shapes, tp_config, mesh)`。
  比较每个被 TP plan 覆盖的参数 local shape 与并行前 reference shape。mesh
  的 `tensor` size 作为 `tp_size`。`"col"` 检查 col 切分维度，分母是
  `tp_size`。`"row"` 检查 row 切分维度，如果同时启用 FSDP2 sharding，分母是 `tp_size * fsdp_shard_size`。
  如果参数是 DTensor，用 `param.to_local().shape` 比较；否则用 `param.shape`。
  当 TP plan 覆盖的所有参数 local shape 都符合预期时，可视为 effectiveness 测试通过。
- 优先先把 `tests/test_structure_template.py`、
  `tests/test_runtime_template.py`、`tests/test_smoke_template.py` 复制到
  recipe-local skill 目录，再只改 import 区块和 TP 相关断言或 launcher 路径。
- 由于这个 skill 往往需要分布式 smoke，复制出来的 `test_smoke.py` 应使用
  `tests/test_smoke_template.py` 里的 `multi_rank_distributed_env(...)`，并把
  运行模式配置为 Tensor Parallel；如果该 skill 路径或用户偏好要求，也可以和
  DDP / FSDP2 shard 组合。
- `test_smoke.py` 必须走该 skill 的完整真实能力路径：真实 engine、真实 recipe
  入口、真实 TP / launcher / logger / checkpoint 接线；禁止用 monkeypatch、fake
  wrapper、fake `parallelize_model`、fake process group、fake device mesh 等
  测试桩把要验证的并行路径短路掉。
- 如果该 recipe 的 full-capability single-step 只能在多卡或 GPU / 分布式环境下
  成立，就把 smoke test 写成真实 launcher 测试，并在 `test_spec.yaml` 里把
  `gpu_preferred` 设为 `true`；不要为了在 CPU 或单进程下跑通而退化成 fake 逻辑。

不要换成与该 recipe 无关的 tiny model。smoke 测试应基于用户自己的 recipe / model
真实入口，只把配置和 batch 缩到该 recipe 能接受的最小规模。

当你在用户 recipe 上执行这个 skill 时，应默认自动补齐这些测试，不要要求用户自己列出测试文件。
验证必须且只能交给全新的 subagent，并使用 `fork_context=false`。禁止主 agent
在本地终端、后台终端会话或其他任何非 subagent shell fallback 中直接运行这些
`python -m tests.test_skills` 命令。先启动一个 subagent 运行
`python -m tests.test_skills --recipe <recipe> --skill tensor-parallel --layer structure`，
只有它通过后，主 agent 才再启动新的 subagent 运行 `--layer runtime`；只有
runtime 通过后，主 agent 才再启动新的 subagent 运行 `--layer smoke`；只有
smoke 通过后，再启动新的 subagent 运行 `--layer effectiveness`。最后由
主 agent 统一汇总四个层级的结果。如果 `test_smoke.py` 或
`test_effectiveness.py` 因 GPU、分布式启动条件或执行权限受限而无法运行，主 agent
直接把准确的 `python -m tests.test_skills` 命令以及所需 launcher 命令返回给用户。

## Output

- 说明更新了哪些 modeling 和 config 文件。
- 按模块类总结 TP plan。
- 说明是否新增了 TP 后处理，以及作用于哪些运行时类。
- 说明最终 mesh 设置，或还缺哪些用户输入才能完成配置。

## Read On Demand

- 需要完整 TP 样例时，读取 `./references/vit_classification/`，其中包含模型改动、配置接线和 recipe-local 测试。
