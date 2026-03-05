---
name: gradient-checkpointing
description: Add gradient checkpointing (activation checkpointing) support to any model in this repo. Use when creating a new model, adding gradient checkpointing to an existing model, or when the user mentions gradient checkpointing, activation checkpointing, or memory optimization for training.
---

# Gradient Checkpointing

为任意模型添加 gradient checkpointing（activation checkpointing）支持。核心思路：**不要试图写一个通用封装函数**，而是针对每个模型的 Encoder 的 forward 逻辑做最小化适配。  
**English:** [SKILL.md](SKILL.md)

## 核心概念

Gradient checkpointing 用计算换显存：前向时不保存中间激活，反向时重新计算。适配的核心就是在 Encoder 的逐层循环里，把每一层的 forward 调用包进 `torch.utils.checkpoint.checkpoint`。

## 适配工作流

对每个新模型，完成以下 3 步：

### Step 1: 在 Encoder 中添加 checkpointing 状态

在 Encoder（包含逐层循环的模块）的 `__init__` 中加两个属性：

```python
self.gradient_checkpointing = False
self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
```

### Step 2: 在 Encoder.forward 中包装每层调用

找到逐层 for 循环，在循环内判断是否启用 checkpointing。**关键模式**：定义一个 `custom_forward` 闭包，只接收需要梯度的 tensor 作为参数，其余参数通过闭包捕获。

```python
use_gc = self.gradient_checkpointing and self.training

for layer in self.layers:
    if use_gc:
        def custom_forward(hidden_states):
            return layer(hidden_states, attention_mask=attention_mask, ...)[0]

        hidden_states = self._gradient_checkpointing_func(custom_forward, hidden_states)
    else:
        hidden_states = layer(hidden_states, attention_mask=attention_mask, ...)[0]
```

**规则**：
- `custom_forward` 的显式参数只放需要 `requires_grad=True` 的 tensor（如 `hidden_states`）。
- 不需要梯度的参数（`attention_mask`、`rotary_pos_emb` 等）通过闭包捕获。
- checkpointing 开启时必须关闭 `output_attentions`（attention weights 不会被保存）。
- 条件：`self.gradient_checkpointing and self.training`（eval 时不 checkpoint）。

**若模型有多种 layer 类型**（如 regular layer + mixture layer），为每种 layer 分别写一个 checkpointing 分支或 helper。参见 [references/example-tomatovit.zh-CN.md](references/example-tomatovit.zh-CN.md) 中的 `_forward_single_branch_layer` 与 `_forward_mixture_layer`。

### Step 3: 在 Model 顶层暴露 enable/disable 接口

两种方式二选一：

**方式 A — HuggingFace PreTrainedModel（推荐）**  
若模型继承 `PreTrainedModel`，只需设置类属性：

```python
class MyPreTrainedModel(PreTrainedModel):
    supports_gradient_checkpointing = True
```

`PreTrainedModel.gradient_checkpointing_enable()` 会自动遍历子模块，设置 `gradient_checkpointing = True` 和 `_gradient_checkpointing_func`。

**方式 B — 纯 nn.Module**  
手动实现 enable/disable：

```python
class MyModel(nn.Module):
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        gc_kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        self.encoder.gradient_checkpointing = True
        self.encoder._gradient_checkpointing_func = functools.partial(
            torch.utils.checkpoint.checkpoint, **gc_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing = False
```

### 在 Engine 中启用

在 recipe engine 的 `prepare_model()` 中，freeze 之后、FSDP/DDP 之前调用：

```python
gc_enabled = OmegaConf.select(self.config, "model.gradient_checkpointing.enabled", default=False)
gc_use_reentrant = OmegaConf.select(self.config, "model.gradient_checkpointing.use_reentrant", default=False)
if gc_enabled:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc_use_reentrant}
    )
```

YAML 配置示例：

```yaml
model:
  gradient_checkpointing:
    enabled: true
    # use_reentrant: false  # 默认 false，一般不需显式设置
```

## 常见陷阱

1. **`use_reentrant=False`**（推荐默认）：支持非确定性操作和 `torch.compile`，但要求 checkpoint 函数的输出对输入可微。
2. **不要在 `custom_forward` 的显式参数中传入不需要梯度的 tensor**：会导致多余重算或报错。
3. **`output_attentions` 与 checkpointing 冲突**：重算时不保存中间结果，attention weights 会丢失，因此 checkpointing 启用时应强制 `output_attentions=False`。
4. **顺序**：freeze → gradient checkpointing → FSDP/DDP wrap。

## 测试

为每个模型写 3 个测试。完整模板见 [references/test-patterns.zh-CN.md](references/test-patterns.zh-CN.md)。

摘要：
1. `test_gradient_checkpointing_enable_sets_state` — enable/disable 正确设置 encoder 状态。
2. `test_encoder_uses_checkpointing` — 训练时确实调用了 checkpoint 函数。
3. `test_gradient_matches_without_checkpointing` — 开启/关闭 checkpointing 时梯度数值一致。

## 完整参考

- TomatoViT 完整适配：[references/example-tomatovit.zh-CN.md](references/example-tomatovit.zh-CN.md)
- 测试模板与完整用例：[references/test-patterns.zh-CN.md](references/test-patterns.zh-CN.md)
