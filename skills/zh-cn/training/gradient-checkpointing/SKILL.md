---
name: gradient-checkpointing
description: 为本仓库中的 recipe 添加 gradient checkpointing。适用于判断模型是否已原生支持 checkpointing、补齐 recipe 的 config 和 engine 开关，或在模型侧做最小化 checkpoint 适配并补测试。
---

# Gradient Checkpointing

## Goal

- 为目标 recipe 启用 gradient checkpointing，同时不改变模型数学行为。
- 保持实现为 recipe-local，而不是引入仓库级通用包装器。
- 补齐 config、engine 接线和能证明功能真的生效的测试。

## Required Inputs

- 目标 recipe 路径，以及负责构建模型和 engine 的文件。
- 顶层模型类，或持有逐层循环的模块。
- 模型是否已经原生暴露 checkpointing 支持。
- 目标 recipe 的 config 或 schema 文件。
- 可以添加 recipe-local 测试的位置。

## Workflow

### 1. 修改前先判断模型属于哪条路径

- 只要模型已经知道如何对重复 block 做 checkpoint，就优先走“已有支持”路径。
- 只有模型内部还没有接好 checkpointing 时，才走“手动适配”路径。

“已有支持”路径需要同时满足：
- 顶层模型暴露 `gradient_checkpointing_enable()` 和 `gradient_checkpointing_disable()`
- 模型会把 `gradient_checkpointing` 和 `_gradient_checkpointing_func` 传递给真正需要的子模块
- 重复 block 在 `self.gradient_checkpointing and self.training` 为真时，已经会通过 `_gradient_checkpointing_func` 执行

### 2. 已有支持路径：只改 recipe 接线

- 如果模型已经支持 checkpointing，就不要重写模型内部逻辑。
- 在 `prepare_model()` 中，于模型构建完成后、FSDP、DDP 或 TP wrap 前启用 checkpointing：

```python
gc_enabled = self.config.model.gradient_checkpointing.enabled
gc_use_reentrant = self.config.model.gradient_checkpointing.use_reentrant
if gc_enabled:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc_use_reentrant}
    )
```

- 增加配置：

```yaml
model:
  gradient_checkpointing:
    enabled: false
    use_reentrant: false
```

- 在新配置系统下，把 `model.gradient_checkpointing` 加到 recipe 的 schema 或 `ConfigClass`，并在 engine 中通过类型化属性访问读取。
- 除非目标模型明确要求 reentrant checkpointing，否则优先使用 `use_reentrant: false`。

### 3. 手动适配路径：修改持有层循环的模块

- 在拥有重复层循环的模块上增加：

```python
self.gradient_checkpointing = False
self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
```

- 在循环里，训练且启用 checkpointing 时，通过 `_gradient_checkpointing_func` 调用每一层。
- 显式传给 checkpoint 的参数只放带梯度的 tensor；mask、RoPE 输入和其他不需要梯度的值通过闭包捕获。
- 如果被 checkpoint 的层无法安全返回 attention、cache 等辅助输出，就对这些 flag 做门控，或返回一致的精简输出。
- 对 `PreTrainedModel` 子类，设置 `supports_gradient_checkpointing = True`。
- 对纯 `nn.Module`，本地实现 `gradient_checkpointing_enable()` 和 `gradient_checkpointing_disable()`。

示例：

```python
use_gc = self.gradient_checkpointing and self.training and not output_attentions

for layer in self.layers:
    if use_gc:
        def custom_forward(hidden_states):
            return layer(hidden_states, attention_mask=attention_mask, ...)[0]

        hidden_states = self._gradient_checkpointing_func(custom_forward, hidden_states)
    else:
        hidden_states = layer(hidden_states, attention_mask=attention_mask, ...)[0]
```

### 4. 添加 recipe-local 测试

至少覆盖以下内容：
- enable 和 disable 开关会正确设置模块状态
- 训练时确实调用了 checkpoint 函数
- 开启和关闭 checkpointing 时梯度一致

### 5. 验证最终接线

- 确认 checkpointing 是在分布式 wrap 之前启用的。
- 确认 config、engine 接线和测试描述的是同一套功能形态。
- 如果模型已经继承 `GradientCheckpointingLayer` 或类似机制，就不要再手工重复包一层。

## Validation

- 选择的路径与模型真实能力一致。
- recipe 配置暴露了 `model.gradient_checkpointing.enabled` 和 `use_reentrant`。
- checkpointing 在 FSDP、DDP 或 TP wrap 之前启用。
- recipe-local 测试覆盖了开关、调用和梯度一致性。
- 实现没有引入仓库级包装器，也没有把不需要梯度的输入作为显式 checkpoint 参数传入。

## Output

- 说明使用的是哪条路径：已有支持还是手动适配。
- 说明更新了哪些 model、engine、config 和 test 文件。
- 总结运行时如何开启 checkpointing。
- 总结已执行验证和仍未验证的部分。

## Read On Demand

- 需要最小 recipe-local 接线样例时，读取 `references/vit_classification/`。
- 需要具体测试样例时，读取 `references/vit_classification/tests/test_vit_gradient_checkpointing_zh.py`。
