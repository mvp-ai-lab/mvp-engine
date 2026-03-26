---
name: fsdp2-prefetching
description: 为 recipe/model 中的新模型补充 FSDP2 prefetching callable。适用于 FSDP2 wrap 已经就绪，但 forward/backward prefetch 顺序依赖具体模型层级、分支结构和 forward 调用顺序的场景。
---

# FSDP2 Prefetching（中文）

## Goal

为目标模型生成一个 recipe/model-local 的 FSDP2 prefetch setup callable，并把它绑定到顶层模型类的 `FSDP2_PREFETCHING` 类属性上。

运行时契约固定如下：
- 入口在 `mvp_engine/distributed/parallelize.py`
- runtime 只做一件事：在 FSDP2 wrap 完成后读取并调用 `model.__class__.FSDP2_PREFETCHING(model)`
- 不新增 YAML 开关，不设计通用 prefetch DSL

## Required Inputs

- 目标 `modeling_*.py` 或等价模型实现文件
- 训练实际使用的顶层模型类
- FSDP2 已经 wrap 的目标层类型或 `_no_split_modules`
- 目标模型 `forward()` 及关键子模块 `forward()` 的源码
- 是否存在多分支、跨层跳转、混合层或共享层

## Workflow

### 1. 判断这件事是否应该做成 skill

- 如果 prefetch 顺序明显依赖模型层间拓扑、分支切换或自定义执行顺序，做成 skill。
- 如果只是单一通用策略参数，应该进 runtime 代码，而不是 skill。
- 不要为 prefetch 设计新的通用配置语言。

### 2. 收集 prefetch wiring 所需结构

- 找到训练实际使用的顶层模型类。
- 找到 FSDP2 真正包裹的重复计算单元，例如 encoder layer、mixture layer、head。
- 只记录会被 `fully_shard()` 包裹的模块；不要把未包裹模块放进 prefetch 边里。
- 阅读顶层 `forward()` 和关键 block 的 `forward()`，按源码顺序列出一次完整前向执行链。
- 如果存在双支路或 mixture 结构，先把每一层内部顺序写出来，再写层与层之间如何衔接。

### 3. 先画出最小 forward/backward prefetch 边

- 前向边规则：
  - 当前模块执行时，应预取“紧接着即将执行的下一个 FSDP2 模块”。
  - 多分支情况下，边按真实执行顺序串起来，不要假设所有分支并行。
- 反向边规则：
  - 先按前向链逆序思考，再补 `set_modules_to_backward_prefetch()`。
  - 只补真正减少等待的关键边；不要为了“完整”把所有相邻模块都连满。
- 如果模型是纯顺序堆叠，优先生成最简单的 layer[i] -> layer[i+1] 规则。
- 如果模型存在分支切换，优先写显式索引或显式列表，不要过度抽象成通用图算法。

### 4. 修改 modeling 代码

- 在模型文件中新增一个最小 callable，例如：
  ```python
  def setup_<model_name>_fsdp2_prefetching(model: nn.Module) -> None:
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
      FSDP2_PREFETCHING = setup_<model_name>_fsdp2_prefetching
  ```
- 如果 modeling 文件里已经存在训练实际使用的顶层 wrapper class，只能在这个已有类上追加 `FSDP2_PREFETCHING`，禁止再创建第二个同名 wrapper class。
- 如果模型同时需要 TP 与 FSDP2 prefetching，必须把 `FSDP2_PREFETCHING`、`TP_MODULE_CONFIG` 和 `TP_MODULE_POSTPROCESSORS` 合并到同一个顶层模型类声明中。
- callable 必须是 recipe/model-local 代码，不要塞回 `mvp_engine/`。
- callable 必须直接从 `model` 上拿到已经 wrap 完的模块实例，不要重新构建模块列表副本。
- 用一个幂等 guard，例如 `_fsdp2_prefetching_configured`，避免重复设置。

### 5. 保持实现简单

- 不要引入 `torch.fx`、trace helper 或自动图分析。
- 不要把模型局部执行顺序抽象成通用 runtime helper。
- 不要修改模型 config 来表达 prefetch 边。
- 如果只是 2-3 类模块的固定 wiring，用显式 for-loop 和分支即可。

## Validation

- 确认顶层模型类定义了 `FSDP2_PREFETCHING`，且其值为 callable。
- 确认若顶层 wrapper class 已存在，本次修改是在原类上追加属性，而不是新建第二个同名类。
- 确认若模型同时启用 TP 与 FSDP2 prefetching，相关类属性已经合并到同一个顶层模型类声明中。
- 确认 callable 在读取模块时依赖的是运行时真实模块路径，而不是猜测的名字。
- 确认所有被加入 prefetch 边的模块都已经进入 FSDP2 wrap 集合。
- 确认 callable 具备幂等 guard，重复调用不会重复改写状态。
- 确认没有引入新的通用 prefetch DSL、graph helper 或 YAML 配置项。
- 至少补一个测试：
  - 轻量单测：验证 callable 被 runtime 调用
  - 或 smoke test：验证并行化后能正常跑通一次 forward/backward

## Output

- 说明新增或修改了哪个模型文件，以及绑定了哪个 `FSDP2_PREFETCHING` callable。
- 说明 forward/backward prefetch 的核心边是如何组织的。
- 说明验证方式和未覆盖的风险。

## Read On Demand

- 当你需要一个顺序堆叠模型的 FSDP2 prefetching 参考实现时，读取 `./references/vit_classification/model/vit.py`。
