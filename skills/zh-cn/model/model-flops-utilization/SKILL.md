---
name: model-flops-utilization
description: 为当前模型和 engine 真正实现 MFU 支持，包括模型 FLOPs 估算、硬件峰值算力查询、运行时 MFU 计算与日志打印。适用于需要把 MFU 接到真实训练流程里，而不是只停留在 FLOPs 公式说明的场景。
---

# Model FLOPs Utilization

> 中文版：`skills/zh-cn/model/model-flops-utilization/SKILL.md`
> English version: `skills/en/model/model-flops-utilization/SKILL.md`

## Goal

- 用注入方式为当前模型实例添加 `calculate_model_flops(...) -> float`，而不是替换整个模型类。
- 在 engine 中新增或接入 `calculate_mfu(...) -> float`，基于模型 FLOPs、step 时间、单卡峰值算力、精度和 `world_size` 计算真实 MFU。
- 检测当前硬件，从 `references/hardware_peak_flops.csv` 中查峰值算力，并把 `mfu=...` 打印到训练日志里。
- 如果当前环境只有 CPU，先在工作目录的 `*.md` 文件里查 GPU / 集群训练 instruction；如果找不到，再询问 user。

## 参考

- 如果 `references/recipes/` 下存在 MFU 示例，请先把它当作“实现样例”来看，先理解通常需要改哪些文件和哪一层代码，再开始改当前 recipe。
- 这个样例只用于识别实现模式，不代表当前模型、字段名、配置层级或训练流程一定相同。
- 对比样例和当前 recipe 时，优先关注这些问题：
  - 模型 FLOPs 是如何挂到模型实例上的
  - MFU 配置是如何进入 schema 或 config 的
  - 训练 step 完成后在哪里计算并打印 `mfu`
  - `world_size`、step 时间、硬件峰值算力分别从哪里获取
- 只有在和当前 recipe 兼容时，才复用样例里的类名、文件名或字段名；否则应该按当前模型和 engine 结构调整。
- 如果样例和当前 recipe 不一致，优先抽象出“改动位置”和“数据流”，不要机械复制实现。

## Required Inputs

- 创建运行时模型实例的入口位置。
- 能拿到 step 时间与 logger 的 engine 或训练循环位置。
- 当前模型的真实架构：ViT、Decoder-only Transformer，或由视觉分支与语言分支组合而成的 VLM。
- MFU 计算需要的运行时信息：
  - 训练精度，例如 `bf16`、`fp16`、`fp32`
  - `world_size`
  - step 时间（秒）或 engine 里已有的等价计时来源
  - 当前设备名称，来自 `nvidia-smi` 或 user 提供
- 当前工作目录，因为 CPU-only 分流需要扫描本地 `*.md` 文件。

## Workflow

### 1. 先确认运行环境

- 先判断当前训练是否已经在 GPU 上运行。
- 如果 GPU 可用，用 `nvidia-smi` 识别当前设备名称，后续据此查峰值算力。
- 如果当前环境是 CPU-only：
  - 扫描当前工作目录下的 `*.md` 文件
  - 查找 GPU、Slurm、cluster、`torchrun` 或其他训练启动 instruction
  - 如果找到了，就 follow 这些 instruction，把流程切到 GPU 环境后再继续
  - 如果没找到，再询问 user 该如何在 GPU 上运行训练
- 只有在硬件身份、精度或启动方式无法从环境中可靠得到时，才询问 user。

### 2. 以注入方式为当前模型添加 `calculate_model_flops(...)`

- 不要把“替换模型类为子类”作为默认方案。
- 增加一个 `inject_model_flops_calculation(model)` helper，用 `types.MethodType` 把 `calculate_model_flops` 动态挂到当前实例上。
- 方法签名只保留当前架构真正需要的最小参数集。
- 不要把无用参数留在签名里。
- 返回单个 `float`：当前进程、当前 step 的模型 FLOPs。

使用这种注入模式：

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, ..., is_training: bool = True) -> float:
        ...

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model


model = AutoModel.from_pretrained(...)
model = inject_model_flops_calculation(model)
```

#### ViT 示例

适用于 patch embedding vision transformer。

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        is_training: bool = True,
    ) -> float:
        if isinstance(image_size, int):
            image_h, image_w = image_size, image_size
        else:
            image_h, image_w = map(int, image_size)
        if isinstance(patch_size, int):
            patch_h, patch_w = patch_size, patch_size
        else:
            patch_h, patch_w = map(int, patch_size)

        batch = int(batch_size)
        if min(batch, image_h, image_w, patch_h, patch_w) <= 0:
            raise ValueError("batch_size, image_size, and patch_size must be > 0")
        if image_h % patch_h != 0 or image_w % patch_w != 0:
            raise ValueError("image_size must be divisible by patch_size")

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        channels = int(getattr(self.config, "num_channels", 3))
        hidden = int(self.config.hidden_size)
        layers = int(self.config.num_hidden_layers)
        intermediate = int(self.config.intermediate_size)
        num_labels = int(getattr(self.config, "num_labels", 1000))

        patch_embed_flops = 2 * batch * num_patches * (channels * patch_h * patch_w) * hidden
        block_flops = (
            8 * batch * num_patches * hidden * hidden
            + 4 * batch * num_patches * num_patches * hidden
            + 4 * batch * num_patches * hidden * intermediate
        )
        head_flops = 2 * batch * hidden * num_labels
        forward_flops = float(patch_embed_flops + layers * block_flops + head_flops)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### Decoder-only Transformer 示例

适用于 dense decoder-only language model。

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        seq_len: int,
        is_training: bool = True,
    ) -> float:
        batch = int(batch_size)
        tokens = int(seq_len)
        if batch <= 0 or tokens <= 0:
            raise ValueError("batch_size and seq_len must be > 0")

        layers = int(getattr(self.config, "num_hidden_layers", self.config.n_layer))
        hidden = int(getattr(self.config, "hidden_size", self.config.n_embd))
        intermediate = int(getattr(self.config, "intermediate_size", 4 * hidden))
        vocab = int(self.config.vocab_size)

        per_layer = (
            8 * batch * tokens * hidden * hidden
            + 4 * batch * tokens * tokens * hidden
            + 4 * batch * tokens * hidden * intermediate
        )
        lm_head_flops = 2 * batch * tokens * hidden * vocab
        forward_flops = float(layers * per_layer + lm_head_flops)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### VLM 示例

适用于运行时模型同时包含 vision encoder 和 decoder-only language model 的情况。不要假设所有 VLM 的字段名都一样。

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        seq_len: int,
        is_training: bool = True,
    ) -> float:
        vision_flops = ...
        language_flops = ...
        forward_flops = float(vision_flops + language_flops)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

- 对 VLM，视觉侧和语言侧都要计入。
- `vision_config`、`text_config`、`hidden_size`、`num_hidden_layers` 这类字段名必须按真实实现调整，不要机械照抄模板。

### 3. 在 engine 中实现真正的 MFU 计算

- 在 engine 侧新增或复用一个 `calculate_mfu(...)` helper。
- MFU 必须按运行时吞吐计算，不能只用模型 FLOPs 静态比一下。
- 优先复用 engine 已有的 step 计时来源；如果 engine 已经在记录 iteration duration，不要再平行造一个计时器。

默认公式写成：

```python
def calculate_mfu(
    *,
    model_flops_per_step: float,
    step_time_seconds: float,
    device_peak_tflops: float,
    world_size: int,
) -> float:
    if step_time_seconds <= 0:
        raise ValueError("step_time_seconds must be > 0")
    if device_peak_tflops <= 0:
        raise ValueError("device_peak_tflops must be > 0")
    if world_size <= 0:
        raise ValueError("world_size must be > 0")

    total_peak_flops = device_peak_tflops * 1e12 * world_size
    achieved_flops_per_second = model_flops_per_step / step_time_seconds
    return float(achieved_flops_per_second / total_peak_flops)
```

- `model_flops_per_step` 使用当前进程视角的模型 FLOPs。
- `device_peak_tflops` 是单卡、当前精度下的峰值算力。
- `world_size` 必须反映真正参与训练 step 的设备数。
- 如果当前 engine 的 step 定义受 gradient accumulation、pipeline parallel 等影响，就按真实 step 边界调整分子和计时来源，并在代码注释里写清楚假设。

### 4. 解析当前硬件的峰值算力

- 优先用 `nvidia-smi` 识别当前 GPU。
- 将识别出的设备名规范化后，到 `references/hardware_peak_flops.csv` 中匹配最接近的一行。
- 再按当前精度，例如 `bf16` 或 `fp16`，选取对应峰值。
- 如果无法可靠匹配，就询问 user 当前用的硬件和精度。
- 查表逻辑要简单直接，不要静默 fallback 到错误的 GPU 行。

### 5. 把 MFU 打印到训练日志

- 在 engine 已经能访问 step 时间和 logger 的位置计算 MFU。
- 把 MFU 放到训练日志里，与其他 step 指标一起打印。
- metric 名称保持显式：`mfu`。

例如：

```python
mfu = self.calculate_mfu(
    model_flops_per_step=model_flops,
    step_time_seconds=step_time,
    device_peak_tflops=device_peak_tflops,
    world_size=world_size,
)
logger.info("step=%s loss=%.4f mfu=%.4f", step, loss, mfu)
```

## Validation

- 确认注入后当前模型实例暴露 `calculate_model_flops(...)`。
- 确认方法签名只包含与当前架构相关的参数。
- 确认 engine 的 `mfu` 计算同时使用了模型 FLOPs、step 时间、硬件峰值算力和 `world_size`。
- 确认 `calculate_model_flops(...)` 和 `calculate_mfu(...)` 都返回 `float`。
- 确认训练日志里能看到 `mfu=...`。

### 6. 最终验收清单

- 当前模型实例已经具备 `calculate_model_flops(...)`，且该方法是通过注入挂到实例上的，而不是通过替换整个模型类得到的。
- `calculate_model_flops(...)` 的参数只包含当前架构真正需要的最小参数集。
- engine 中已经存在 `calculate_mfu(...)` 或等价的 MFU 计算接入点。
- MFU 的计算同时使用了运行时 step 时间、单卡峰值算力、当前精度和 `world_size`。
- 日志中的 MFU 值是 `float`。
- 训练日志中可以直接看到 `mfu=...`。
- 日志中的 MFU 值通过基本 sanity check：
  - 不是负数
  - 不会明显大于 `1`
  - 如果值可疑，需要回头检查模型 FLOPs、硬件查表、精度映射、step 时间来源和 `world_size`

## Output

- 说明 `calculate_model_flops(...)` 被注入到了哪里。
- 说明 engine 里的 MFU 计算被加到了哪里，或者接入到了哪个现有位置。
- 说明峰值算力查表使用了哪张 GPU、哪种精度。
- 说明当前按单卡还是多卡假设计算。
- 说明 `mfu=...` 会出现在什么日志位置。
- 如果当前没有 GPU 且本地 `*.md` 中也没有启动 instruction，要明确说明还需要 user 提供 GPU 运行方式，MFU 才能安全完成。

## Read On Demand

- 需要查常见 GPU 在不同精度下的峰值算力时，读 `references/hardware_peak_flops.csv`。
