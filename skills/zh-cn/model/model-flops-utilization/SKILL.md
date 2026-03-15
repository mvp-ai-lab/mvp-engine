---
name: model-flops-utilization
description: 为当前模型新增或修正 `calculate_model_flops(...)`，并验证该方法是否可调用、是否满足返回协议、以及在 Transformer 和 ViT 架构上的泛化方式。适用于需要真正实现 MFU 支持，而不是只写公式说明的场景。
---

# Model FLOPs Utilization

为目标模型真正实现 `calculate_model_flops(...)`。不要停留在公式解释或伪代码。
**English:** [SKILL.md](../../../en/model/model-flops-utilization/SKILL.md)

## 目标

- 在当前模型或它的本地适配类上新增 `calculate_model_flops(...) -> float`。
- 使用显式形状参数，而不是依赖隐式运行时状态。
- 验证该方法可以被调用、返回 `float`，并正确处理边界输入。
- 在 `references/` 中保留 Transformer 与 ViT 两类可复用示例。

## 1. 修改前先判断目标模型属于哪条路径

- 如果模型类定义在当前仓库并且由当前 recipe 直接持有，优先走“原地实现”路径。
- 如果运行时模型类来自 `transformers` 等三方包，走“本地适配类”路径。
- 选择与真实重复计算结构一致的架构模板，不要硬套错误公式。

### 原地实现路径

满足以下条件时走这条路径：

- 模型类定义在本仓库内。
- recipe 直接实例化这个类。
- 只加一个方法不会引入额外大规模重构。

### 本地适配类路径

满足以下任一条件时走这条路径：

- 模型类来自第三方包。
- 不适合修改 vendor 源码。
- recipe 已经有本地薄封装，或者能安全替换成一个本地子类。

对这条路径，定义一个本地子类，在子类中实现 `calculate_model_flops(...)`，并把实例化位置切换到该子类。

## 2. 交付的是方法实现，不只是公式

每次正确使用这个 skill，都必须产出一个可运行的方法，满足以下统一契约：

```python
def calculate_model_flops(
    self,
    *,
    batch_size: int,
    seq_len: int | None = None,
    image_size: int | tuple[int, int] | None = None,
    patch_size: int | tuple[int, int] | None = None,
    is_training: bool = True,
) -> float:
    ...
```

规则：

- Transformer：必须显式接收 `batch_size`、`seq_len`、`is_training`。
- ViT：必须显式接收 `batch_size`、`image_size`、`patch_size`、`is_training`。
- 缺失架构必需输入时，抛出 `ValueError`。
- 返回单个 `float`：单进程每 step FLOPs。
- 如果内部为了调试计算了 breakdown dict，也不要把它替代 `float` 返回。

## 3. 选择匹配的实现模板

### Transformer 模板

适用于 dense encoder、decoder-only、encoder-decoder 这类 FLOPs 主要由 attention 和 MLP block 决定的模型。

```python
def calculate_model_flops(
    self,
    *,
    batch_size: int,
    seq_len: int | None = None,
    image_size: int | tuple[int, int] | None = None,
    patch_size: int | tuple[int, int] | None = None,
    is_training: bool = True,
) -> float:
    if seq_len is None:
        raise ValueError("Transformer FLOPs requires seq_len.")

    B = int(batch_size)
    S = int(seq_len)
    L = int(self.config.num_hidden_layers)
    H = int(self.config.hidden_size)
    I = int(self.config.intermediate_size)

    per_layer = 8 * B * S * H * H + 4 * B * S * S * H + 4 * B * S * H * I
    transformer_flops = L * per_layer

    lm_head_flops = 0.0
    if hasattr(self.config, "vocab_size"):
        V = int(self.config.vocab_size)
        lm_head_flops = 2 * B * S * H * V

    forward_flops = float(transformer_flops + lm_head_flops)
    return forward_flops * 3.0 if is_training else forward_flops
```

### ViT 模板

适用于 patch embedding + transformer block + classification head 结构的 Vision Transformer。

```python
def calculate_model_flops(
    self,
    *,
    batch_size: int,
    seq_len: int | None = None,
    image_size: int | tuple[int, int] | None = None,
    patch_size: int | tuple[int, int] | None = None,
    is_training: bool = True,
) -> float:
    if image_size is None or patch_size is None:
        raise ValueError("ViT FLOPs requires image_size and patch_size.")

    B = int(batch_size)
    if isinstance(image_size, int):
        img_h, img_w = image_size, image_size
    else:
        img_h, img_w = map(int, image_size)
    if isinstance(patch_size, int):
        p_h, p_w = patch_size, patch_size
    else:
        p_h, p_w = map(int, patch_size)

    if min(B, img_h, img_w, p_h, p_w) <= 0:
        raise ValueError("batch_size, image_size, and patch_size must be > 0")
    if img_h % p_h != 0 or img_w % p_w != 0:
        raise ValueError("image_size must be divisible by patch_size")

    N = (img_h // p_h) * (img_w // p_w)
    C = int(getattr(self.config, "num_channels", 3))
    D = int(self.config.hidden_size)
    L = int(self.config.num_hidden_layers)
    I = int(self.config.intermediate_size)
    K = int(getattr(self.config, "num_labels", 1000))

    patch_embed_flops = 2 * B * N * (C * p_h * p_w) * D
    block_flops = 8 * B * N * D * D + 4 * B * N * N * D + 4 * B * N * D * I
    backbone_flops = L * block_flops
    head_flops = 2 * B * D * K

    forward_flops = float(patch_embed_flops + backbone_flops + head_flops)
    return forward_flops * 3.0 if is_training else forward_flops
```

### 外部模型本地适配模板

```python
class ExternalModelWithFlops(ExternalModel):
    def calculate_model_flops(... ) -> float:
        ...

# 把：
# model = ExternalModel(config)
# 替换为：
# model = ExternalModelWithFlops(config)
```

## 4. 实现后立刻验证

没有实际执行过的方法，不算完成。

验证清单：

1. 目标模型实例确实暴露 `calculate_model_flops`。
2. 方法签名包含该架构要求的显式参数。
3. `is_training=True` 与 `is_training=False` 都可以执行。
4. 返回类型是 `float`。
5. FLOPs 为正数。
6. 训练 FLOPs 大于等于推理 FLOPs。
7. 缺失必需形状参数时抛 `ValueError` 或 `TypeError`。

## 5. 示例要归档，不要把验证逻辑散落在外面

把示例实现与测试归档到 `references/`，让 skill 可复用。

本 skill 的参考内容：

- `references/external_vit/`：第三方 ViT 的本地子类接入模式。
- `references/decoder_transformer/`：decoder 风格 Transformer 的直接方法实现模式。
- `references/validation_cases.py`：用于验证触发与合同的提示词集合。
- `references/run_validation.py`：dry-run 模板生成脚本。
- `references/check_acceptance.py`：验收门槛计算脚本。

## 常见陷阱

- 用户明确要“实现”时，不要只返回公式解释。
- 不要只返回 dict；主协议必须是单个 `float`。
- 不要把必需形状参数藏到运行时 tensor 里推断；skill 明确要求显式参数。
- 对第三方模型，不要直接改包源码；优先本地子类。
- 不要跳过执行验证；方法存在但跑不通，仍然算失败。
- 不要假设一套公式天然覆盖 MoE、稀疏注意力、fused kernel、activation recomputation；这些要作为边界条件写清楚。

## 参考

- 外部 ViT 示例：`references/external_vit/`
- Decoder Transformer 示例：`references/decoder_transformer/`
- 验证提示词：`references/validation_cases.py`
