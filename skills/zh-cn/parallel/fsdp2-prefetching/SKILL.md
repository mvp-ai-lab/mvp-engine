---
name: fsdp2-prefetching
description: 为 recipe 或 model 中的新模型补充 FSDP2 prefetching callable。适用于
  FSDP2 wrapping 已经就绪，但 forward 或 backward prefetch 顺序依赖具体模型层级、
  分支结构和 forward 调用顺序的场景。
---

# FSDP2 Prefetching

## Goal

- 为目标模型生成一个 recipe 或 model-local 的 FSDP2 prefetch setup callable。
- 把这个 callable 绑定到顶层模型类的 `APPLY_FSDP2_CUSTOM_PREFETCHING` 属性上。
- 保持实现留在 recipe 或模型局部，不引入通用 prefetch DSL 或运行时 helper。

本仓库的运行时契约固定如下：

- 入口在 `mvp_engine/distributed/fsdp2.py`
- runtime 只会在 FSDP2 wrap 完成后读取并调用
  `model.__class__.APPLY_FSDP2_CUSTOM_PREFETCHING(model)`
- 不新增 YAML 开关，不设计通用 prefetch DSL

## Required Inputs

- 目标 `modeling_*.py` 或等价模型实现文件
- 训练实际使用的顶层模型类
- FSDP2 wrap 目标或 `_no_split_modules`
- 顶层 `forward()` 与关键子模块 `forward()` 的源码
- 模型是否包含多分支、跨层跳转、mixture 层或共享 block

## Workflow

### 1. 先判断是否真的需要 custom prefetching

- 这个 skill 主要用于真实执行路径不是简单线性堆叠的模型，例如多分支结构、
  跨层切换、mixture 层，或其他默认 FSDP2 overlap 很难贴合执行顺序的情况。
- 如果用户已经观察到 wrapped module 之间明显等待、branch handoff 空转，或通信与
  计算重叠不足，就可以考虑增加 custom prefetching。
- 如果模型基本是顺序堆叠，默认 FSDP2 行为通常已经够用，除非用户明确要继续做性能优化。

### 2. 收集 prefetch wiring 所需结构

- 找到训练实际使用的顶层模型类。
- 找到 FSDP2 真正包裹的重复计算单元，例如 encoder layer、mixture layer、head。
- 只记录会被 `fully_shard()` 包裹的模块；不要把未包裹模块放进 prefetch 边里。
- 阅读顶层 `forward()` 和关键 block 的 `forward()`，按源码顺序列出完整的前向执行链。
- 如果存在多分支或 mixture 结构，先写清单层内部顺序，再写层与层之间如何衔接。

### 3. 先画出最小 forward 和 backward prefetch 边

- 前向边规则：
  - 当前模块执行时，应预取紧接着即将执行的下一个 FSDP2 模块。
  - 多分支情况下，边要遵循真实执行顺序，不要假设所有分支并行。
- 反向边规则：
  - 先按前向链逆序思考，再补 `set_modules_to_backward_prefetch()`。
  - 只补真正能减少等待的关键边；不要为了“完整”把所有相邻模块都连满。
- 如果模型是纯顺序堆叠，优先生成最简单的 `layer[i] -> layer[i + 1]` 规则。
- 如果模型存在分支切换，优先用显式索引或显式列表，不要过度抽象成通用图算法。

### 4. 修改 modeling 代码

- 在模型文件中新增一个最小 callable，例如：

```python
def apply_fsdp2_custom_prefetching_for_<model_name>(model: nn.Module) -> None:
    if getattr(model, "_fsdp2_prefetching_configured", False):
        return
    ...
    layer_a.set_modules_to_forward_prefetch([layer_b])
    layer_b.set_modules_to_backward_prefetch([layer_a])
    model._fsdp2_prefetching_configured = True
```

- 然后把它绑定到顶层模型类：

```python
class <TopModelClass>(...):
    APPLY_FSDP2_CUSTOM_PREFETCHING = apply_fsdp2_custom_prefetching_for_<model_name>
```

- 如果 modeling 文件里已经存在训练实际使用的顶层 wrapper class，只能在这个已有类上
  追加 `APPLY_FSDP2_CUSTOM_PREFETCHING`，不要再创建第二个同名 wrapper class。
- 如果模型同时需要 TP 与 FSDP2 prefetching，必须把
  `APPLY_FSDP2_CUSTOM_PREFETCHING`、`TP_MODULE_CONFIG` 和
  `TP_MODULE_POSTPROCESSORS` 合并到同一个顶层模型类声明中。
- callable 必须保持为 recipe 或 model-local 代码，不要移到 `mvp_engine/`。
- callable 必须直接从 `model` 上读取已经 wrap 完的模块实例，不要重新构建影子模块列表。
- 用 `_fsdp2_prefetching_configured` 这类幂等 guard 避免重复设置。

### 5. 保持实现简单

- 不要引入 `torch.fx`、trace helper 或自动图分析。
- 不要把模型局部执行顺序抽象成通用运行时 helper。
- 不要修改模型 config 来表达 prefetch 边。
- 如果 wiring 只涉及少数几类模块，就用显式循环和分支。

## Validation

- 确认顶层模型类定义了 `APPLY_FSDP2_CUSTOM_PREFETCHING`，且其值为 callable。
- 确认若顶层 wrapper class 已存在，本次修改是在原类上追加属性，而不是新建第二个同名类。
- 确认若模型同时启用 TP 与 FSDP2 prefetching，相关类属性已经合并到同一个顶层模型类声明中。
- 确认 callable 在读取模块时依赖的是运行时真实模块路径，而不是猜测的名字。
- 确认所有被加入 prefetch 边的模块都已经进入 FSDP2 wrap 集合。
- 确认 callable 具备幂等 guard，重复调用不会重复改写状态。
- 确认没有引入新的通用 prefetch DSL、graph helper 或 YAML 配置项。

在 `recipes/<recipe>/skill_tests/fsdp2-prefetching/` 下补 recipe-local 测试：

- `test_spec.yaml`：声明这个 skill 在该 recipe 上要求哪些测试层级。
- `test_structure.py`：至少验证 recipe import、registry 接线、config schema 校验、
  required slots，以及 logger/checkpoint hooks；还必须验证顶层模型类暴露了
  `APPLY_FSDP2_CUSTOM_PREFETCHING`。
- `test_runtime.py`：至少成功构建 dataset、collator、model、optimizer、
  scheduler 和 engine，且不直接启动训练；还必须验证运行时确实调用 hook，
  且幂等保护生效。
- `test_smoke.py`：覆盖 1 个真实、recipe-owned 的 single step：forward、loss、
  backward、optimizer step、logger write，以及 checkpoint noop 或临时保存；
  还必须验证应用该 hook 后，用户自己的 recipe / model 能在 FSDP2 prefetching
  开启的情况下完成这一步。
- 优先先把 `tests/test_structure_template.py`、
  `tests/test_runtime_template.py`、`tests/test_smoke_template.py` 复制到
  recipe-local skill 目录，再只改 import 区块和 FSDP2 prefetch 相关断言或
  launcher 路径。
- 由于这个 skill 往往需要分布式 smoke，复制出来的 `test_smoke.py` 应使用
  `tests/test_smoke_template.py` 里的 `multi_rank_distributed_env(...)`，并把
  运行模式配置为 FSDP2 shard；如果该 skill 路径或用户偏好要求，也可以和 DDP
  / Tensor Parallel 组合。
- `test_smoke.py` 必须走完整真实能力路径：真实 engine、真实 parallelize 入口、
  真实 FSDP2 wrap / TP / launcher / logger / checkpoint；禁止用 monkeypatch、fake
  wrapper、fake parallelize_model、fake fully_shard、fake process group、fake
  device mesh 等方式把并行能力短路掉。
- 如果该 recipe 的 full-capability single-step 只能在多卡或 GPU 环境下成立，就把
  smoke test 写成真实 launcher 测试，并在 `test_spec.yaml` 里把 `gpu_preferred`
  设为 `true`；不要为了在 CPU 或单进程下跑通而退化成 fake 逻辑。

这些测试必须围绕用户自己的 recipe / model 落点来写，不能为了方便而换成无关的 tiny model。

当你在用户 recipe 上执行这个 skill 时，应默认自动补齐这些测试，不要等用户再单独提出。
验证必须且只能交给全新的 subagent，并使用 `fork_context=false`。禁止主 agent
在本地终端、后台终端会话或其他任何非 subagent shell fallback 中直接运行这些
`python -m tests.test_skills` 命令。先启动一个 subagent 运行
`python -m tests.test_skills --recipe <recipe> --skill fsdp2-prefetching --layer structure`，
只有它通过后，主 agent 才再启动新的 subagent 运行 `--layer runtime`；只有
runtime 通过后，主 agent 才再启动新的 subagent 运行 `--layer smoke`。最后由
主 agent 统一汇总三个层级的结果。如果 `test_smoke.py` 因 GPU、分布式启动条件
或执行权限受限而无法运行，主 agent 直接把准确的 `python -m tests.test_skills`
命令以及所需 launcher 命令返回给用户。

## Output

- 说明新增或修改了哪个模型文件，以及绑定了哪个
  `APPLY_FSDP2_CUSTOM_PREFETCHING` callable。
- 说明 forward 和 backward prefetch 的核心边。
- 说明已验证内容和仍未验证的部分。

## Read On Demand

- 需要顺序堆叠模型的 FSDP2 prefetching 参考实现时，读取
  `./references/vit_classification/model/vit.py`。
