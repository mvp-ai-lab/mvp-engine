---
name: gradient-checkpointing
description: 为本仓库中的 recipe 添加 gradient checkpointing（activation checkpointing）。适用于判断模型是否已原生支持 checkpointing、补齐 recipe 的 config 和 engine 开关，或在模型侧做最小化 checkpoint 适配并补测试。
---

# Gradient Checkpointing

添加 gradient checkpointing，但不要为此引入仓库级通用包装器。  
**English:** [SKILL.md](../../../en/training/gradient-checkpointing/SKILL.md)

## 目标

- 为目标 recipe 启用 gradient checkpointing。
- 不改变模型数学行为。
- 补齐 recipe 级 config、engine 接线和验证测试。

## 1. 修改前先判断模型属于哪条路径

- 优先走“已有支持”路径。
- 只有在模型本身还不会对重复 block 做 checkpoint 时，才走“手动适配”路径。

### 已有支持路径

满足以下条件时走这条路径：

- 顶层模型暴露了 `gradient_checkpointing_enable()` 和 `gradient_checkpointing_disable()`。
- 模型会把 `gradient_checkpointing` 和 `_gradient_checkpointing_func` 传递到实际需要的子模块。
- 重复 block 在 `self.gradient_checkpointing and self.training` 为真时，已经会通过 `_gradient_checkpointing_func` 执行。对新版 `transformers` 模型，这通常由 `GradientCheckpointingLayer.__call__` 提供。

### 手动适配路径

当重复计算 block 还不会自行 checkpoint 时，走这条路径。

## 2. 已有支持路径：只改 recipe 接线

- 如果模型已经支持 checkpointing，不要再重写模型内部逻辑。
- 在 `prepare_model()` 中，于模型构建完成后、FSDP/DDP/TP wrap 前启用 checkpointing：

```python
gc_enabled = OmegaConf.select(self.config, "model.gradient_checkpointing.enabled", default=False)
gc_use_reentrant = OmegaConf.select(self.config, "model.gradient_checkpointing.use_reentrant", default=False)
if gc_enabled:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc_use_reentrant}
    )
```

- 配置项增加：

```yaml
model:
  gradient_checkpointing:
    enabled: false
    use_reentrant: false
```

- 除非目标模型明确要求 reentrant checkpointing，否则优先使用 `use_reentrant: false`。

### ViT 参考

- `recipes/vit_classification` 是标准的简单路径。
- HuggingFace ViT 的 layer 已经继承 `GradientCheckpointingLayer`，因此示例只改 recipe 的 engine、config 和测试。
- 归档后的参考文件在 `references/vit_classification/`。

## 3. 手动适配路径：改拥有逐层循环的模块

- 在拥有逐层循环的模块上增加：

```python
self.gradient_checkpointing = False
self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
```

- 在循环里，训练且启用 checkpointing 时，通过 `_gradient_checkpointing_func` 调用每一层。
- 显式传给 checkpoint 的参数只放带梯度的 tensor。mask、rope embedding 等不需要梯度的输入通过闭包捕获。
- 若被 checkpoint 的层无法安全返回 attention、cache 等辅助输出，就对这些 flag 做门控，或返回一套一致的精简输出。
- 对 `PreTrainedModel` 子类，设置 `supports_gradient_checkpointing = True`。
- 对纯 `nn.Module`，本地实现 `gradient_checkpointing_enable()` 与 `gradient_checkpointing_disable()`。

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

## 4. 示例要归档，不要把演示性 recipe 改动直接提交

- 如果你临时修改了 demo recipe 来验证流程，把最终改动后的文件移动到 `references/<recipe>/`。
- 提交 skill 前，把 `recipes/<recipe>/` 恢复到分支的干净状态。
- 只归档那些真正因为 checkpointing 而变动的文件。

## 5. 测试

为 recipe-local 测试至少覆盖：

1. enable/disable 开关能正确设置模块状态。
2. 训练时确实调用了 checkpoint 函数。
3. 开启和关闭 checkpointing 时梯度一致。

参考测试：

- `references/vit_classification/tests/test_vit_gradient_checkpointing.py`

## 常见陷阱

- 不要为此加仓库级通用包装器。
- 不要把不需要梯度的输入作为显式 checkpoint 参数传入。
- 一定要在分布式 wrap 之前启用 checkpointing。
- 对已经使用 `GradientCheckpointingLayer` 的 `transformers` 模型，不要再手工重复包一层。

## 参考

- ViT 的最小 recipe 接入：`references/vit_classification/`
