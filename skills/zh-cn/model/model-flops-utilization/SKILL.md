---
name: model-flops-utilization
description: 为当前模型和 engine 真正实现 MFU 支持，包括模型 FLOPs 估算、硬件峰值算力查询、运行时 MFU 计算与日志记录。适用于需要把 MFU 接到真实训练流程里，而不是只停留在 FLOPs 公式说明的场景。
---

# Model FLOPs Utilization

> 中文版：`skills/zh-cn/model/model-flops-utilization/SKILL.md`
> English version: `skills/en/model/model-flops-utilization/SKILL.md`

## Goal

- 用注入方式为当前模型实例添加 `calculate_model_flops(...) -> float`，不替换模型类。
- 在 engine 中新增或接入 `calculate_mfu(...) -> float`，基于模型 FLOPs、step 时间、单卡峰值算力、精度与 `world_size` 计算 MFU。
- 通过日志字典写入 MFU，键名使用 `perf/mfu`。
- 参考 `references/recipes/vit_classification_addon/` 的 demo 实现路径，再按当前 recipe 结构做适配。

## Required Inputs

- 运行时模型实例创建入口。
- 能拿到 step 时间与 log dict 更新位置的 engine/训练循环代码。
- 真实模型架构：ViT、Decoder-only Transformer、Seq2Seq Encoder-Decoder、或 VLM（视觉+语言）。
- 运行时信息：精度、`world_size`、step 时间、硬件名称。

## Workflow

### 1）先确认运行环境和硬件

- 判断当前是否运行在 GPU。
- 有 GPU 时用 `nvidia-smi` 获取设备名。
- CPU-only 时先扫描本地 `*.md` 中的 GPU/集群启动 instruction；若没有，再询问 user。

### 2）用注入方式给当前模型增加 `calculate_model_flops(...)`

- 实现 `inject_model_flops_calculation(model)`。
- 用 `types.MethodType` 把方法挂到实例。
- 方法签名保持最小必要参数，不留无用参数。

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, batch_size: int, seq_len: int, is_training: bool = True) -> float:
        ...

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

用法：

```python
model = AutoModel.from_pretrained(...)
# model = MyModel()
model = inject_model_flops_calculation(model)
```

#### ViT 示例（不使用 `image_size` / `patch_size` 入参）

从配置或模型内部属性读取图像尺寸与 patch 尺寸，不把它们放进方法签名。

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, batch_size: int, is_training: bool = True) -> float:
        batch = int(batch_size)
        if batch <= 0:
            raise ValueError("batch_size must be > 0")

        cfg = self.config
        hidden = int(cfg.hidden_size)
        layers = int(cfg.num_hidden_layers)
        intermediate = int(cfg.intermediate_size)
        num_labels = int(getattr(cfg, "num_labels", 1000))

        image_size = int(cfg.image_size)
        patch_size = int(cfg.patch_size)
        num_patches = (image_size // patch_size) ** 2
        channels = int(getattr(cfg, "num_channels", 3))

        patch_embed = 2 * batch * num_patches * (channels * patch_size * patch_size) * hidden
        block_flops = (
            8 * batch * num_patches * hidden * hidden
            + 4 * batch * num_patches * num_patches * hidden
            + 4 * batch * num_patches * hidden * intermediate
        )
        head = 2 * batch * hidden * num_labels
        forward_flops = float(patch_embed + layers * block_flops + head)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### Decoder-only Transformer 示例

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, batch_size: int, seq_len: int, is_training: bool = True) -> float:
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
        lm_head = 2 * batch * tokens * hidden * vocab
        forward_flops = float(layers * per_layer + lm_head)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### Seq2Seq Encoder-Decoder 专用模板

单独提供 seq2seq 模板：使用 `encoder_seq_len` 与 `decoder_seq_len`，并显式计入 decoder cross-attention。

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        encoder_seq_len: int,
        decoder_seq_len: int,
        is_training: bool = True,
    ) -> float:
        b = int(batch_size)
        s_enc = int(encoder_seq_len)
        s_dec = int(decoder_seq_len)
        if min(b, s_enc, s_dec) <= 0:
            raise ValueError("batch_size, encoder_seq_len, decoder_seq_len must be > 0")

        cfg = self.config
        hidden = int(cfg.d_model)
        ffn = int(cfg.d_ff)
        enc_layers = int(getattr(cfg, "num_layers", cfg.num_hidden_layers))
        dec_layers = int(getattr(cfg, "num_decoder_layers", enc_layers))
        vocab = int(cfg.vocab_size)

        enc_layer = 8 * b * s_enc * hidden * hidden + 4 * b * s_enc * s_enc * hidden + 4 * b * s_enc * hidden * ffn
        dec_self = 8 * b * s_dec * hidden * hidden + 4 * b * s_dec * s_dec * hidden + 4 * b * s_dec * hidden * ffn
        dec_cross = 4 * b * s_dec * s_enc * hidden
        lm_head = 2 * b * s_dec * hidden * vocab

        forward_flops = float(enc_layers * enc_layer + dec_layers * (dec_self + dec_cross) + lm_head)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### VLM 示例

VLM 需要合并视觉侧与语言侧 FLOPs：

```python
def calculate_model_flops(...):
    vision_flops = ...
    language_flops = ...
    return float((vision_flops + language_flops) * 3.0 if is_training else (vision_flops + language_flops))
```

### 3）在 engine 中计算并记录 MFU

- 增加或复用 `calculate_mfu(...)`。
- 在 engine 已有 step 计时边界计算 MFU。
- 把 MFU 放到 log dict，键为 `perf/mfu`，不要单独 `print`。

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


log_dict = {
    "train/loss": float(loss),
    "perf/step_time": float(step_time_seconds),
    "perf/mfu": float(mfu),
}
logger.log(log_dict, step=global_step)
```

### 4）硬件峰值查表

- 用 `nvidia-smi` 识别设备名。
- 在 `references/hardware_peak_flops.csv` 匹配硬件与精度。
- 无法可靠匹配时询问 user。

### 5）验收

- 注入后 `hasattr(model, "calculate_model_flops")` 为真。
- 方法签名只保留架构必要参数。
- `calculate_model_flops(...)` 与 `calculate_mfu(...)` 返回 `float`。
- 训练日志指标包含 `perf/mfu`。
- MFU 值基本合理（非负，通常不明显大于 1）。
- 若 skill 带路由/契约验证，需同时校验 trigger 与 contract 两类结果。

## Demo 引用路径

说明实现路径时，引用以下 demo 文件：

- `references/recipes/vit_classification_addon/model/vit.py`
- `references/recipes/vit_classification_addon/engine/vit_classification_engine.py`
- `references/recipes/vit_classification_addon/configs/schema.py`
- `references/recipes/vit_classification_addon/configs/train.yaml`

## Output

- 说明 `inject_model_flops_calculation(model)` 在哪里调用。
- 说明 `calculate_mfu(...)` 在 engine 哪里接入。
- 说明峰值查表使用的 GPU 与精度。
- 说明 `world_size` 假设（单卡/多卡）。
- 说明日志里的 `perf/mfu` 在哪里写出。
