---
name: model-flops-utilization
description: 为当前模型和 engine 实现端到端的 MFU 支持，包括模型 FLOPs 估算、硬件峰值算力查询、运行时 MFU 计算和日志记录。
---

# Model FLOPs Utilization

## Goal

- 通过注入而不是替换模型类的方式，为当前模型实例添加 `calculate_model_flops(...) -> float`。
- 在 engine 中新增或接入 `calculate_mfu(...) -> float`，使 MFU 基于运行时 step 时间、模型 FLOPs、当前精度、单卡峰值算力和 `world_size` 计算。
- 把 MFU 作为 `perf/mfu` 写入训练日志。
- 基于真实运行环境和硬件信息实现，不要依赖硬编码假设。

## Required Inputs

- 创建运行时模型实例的入口位置。
- 已经能拿到 step 时间和 logger 的 engine 或训练循环位置。
- 当前模型的真实架构，例如 ViT、decoder-only Transformer、encoder-decoder，或由视觉与语言分支组合而成的 VLM。
- MFU 所需的运行时信息：
  - 当前精度，例如 `bf16`、`fp16`、`fp32`
  - `world_size`
  - step 时间（秒）或 engine 里已有的等价计时来源
  - 当前设备名称，来自 `nvidia-smi` 或用户输入
- 当前工作目录，因为 CPU-only 分流需要扫描本地 `*.md` 查 GPU 启动说明。

## Workflow

### 1. 先看参考实现模式

- 如果 `references/recipes/` 下有 MFU 示例，先把它当成模式样例，而不是可直接照抄的实现。
- 优先关注这些问题：
  - 模型 FLOPs 是怎么挂到运行时模型实例上的
  - MFU 是怎么进入 schema 或 config 的
  - engine 在哪里计算和打印 `mfu`
  - 计时、`world_size` 和峰值算力从哪里获取
- 优先抽取数据流和接入点，而不是机械复用示例里的命名。

### 2. 修改前先确认运行环境

- 先判断当前流程是否已经在 GPU 上运行。
- 如果 GPU 可用，用 `nvidia-smi` 识别当前设备名称，后续据此查峰值算力。
- 如果当前环境是 CPU-only：
  - 扫描本地 `*.md`，查找 GPU、Slurm、cluster、`torchrun` 或其他启动说明
  - 如果找到说明，就按说明切到 GPU 环境后再继续
  - 如果找不到，就停止并询问用户如何启动 GPU 训练
- 只有在硬件身份、精度或启动方式无法可靠推断时，才询问用户。

### 3. 以注入方式为当前模型添加 `calculate_model_flops(...)`

- 默认不要通过子类替换整个模型类。
- 增加 `inject_model_flops_calculation(model)` helper，并用 `types.MethodType` 绑定 `calculate_model_flops`。
- 方法签名只保留当前架构真正需要的最小参数集。
- 返回单个 `float`，表示当前进程视角下每个 step 的模型 FLOPs。

使用这种基础注入模式：

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, ..., is_training: bool = True) -> float:
        ...

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### ViT 示例

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        is_training: bool = True,
    ) -> float:
        batch = int(batch_size)
        if batch <= 0:
            raise ValueError("batch_size must be > 0")

        image_size = int(self.config.image_size)
        patch_size = int(self.config.patch_size)
        num_patches = (image_size // patch_size) ** 2
        channels = int(getattr(self.config, "num_channels", 3))
        hidden = int(self.config.hidden_size)
        layers = int(self.config.num_hidden_layers)
        intermediate = int(self.config.intermediate_size)
        num_labels = int(getattr(self.config, "num_labels", 1000))

        patch_embed_flops = 2 * batch * num_patches * (channels * patch_size * patch_size) * hidden
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

#### Encoder-decoder 示例

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

- 对 VLM，视觉侧和语言侧 FLOPs 都要计入。
- `vision_config`、`text_config`、`hidden_size`、`num_hidden_layers` 等字段名必须按真实实现调整，不要机械照抄模板。

### 4. 在 engine 中实现 MFU

- 新增或复用 engine 侧的 `calculate_mfu(...)` helper。
- 优先复用 engine 已有的 step 计时来源，而不是再平行造一个计时器。
- MFU 必须使用运行时吞吐，而不是只拿静态 FLOPs 做比值。

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

- 如果 gradient accumulation 或 pipeline timing 改变了真实的 step 边界，就按真实边界调整分子和计时来源，并把假设写清楚。

### 5. 显式解析硬件峰值算力

- 优先用 `nvidia-smi` 识别当前 GPU。
- 将设备名规范化后，匹配 `references/hardware_peak_flops.csv` 中最接近的一行。
- 再按当前精度，例如 `bf16` 或 `fp16`，选取对应峰值算力。
- 如果无法可靠匹配，就停止并询问用户当前硬件和精度。
- 查表逻辑必须显式，不能静默 fallback 到错误设备行。

### 6. 把 MFU 写入日志字典

- 在 engine 已经能访问 step 时间和 logger 的位置计算 MFU。
- 用明确的键 `perf/mfu` 记录日志。

```python
mfu = self.calculate_mfu(
    model_flops_per_step=model_flops,
    step_time_seconds=step_time,
    device_peak_tflops=device_peak_tflops,
    world_size=world_size,
)
log_dict["perf/mfu"] = float(mfu)
logger.log(log_dict, step=step)
```

## Validation

- 当前模型实例已经具备 `calculate_model_flops(...)`，且该方法是通过注入得到，而不是替换整个模型类得到。
- 方法签名只包含真实架构需要的参数。
- engine 中存在真实的 MFU 接入点，并且同时使用了 step 时间、峰值算力、当前精度和 `world_size`。
- 日志中的 MFU 值是 `float`，并写在 `perf/mfu` 下。
- MFU 数值通过基本 sanity check：不是负数、不会明显大于 `1`，可疑值已回溯检查 FLOPs 估算、硬件查表、精度映射、计时来源和 `world_size`。

在 `recipes/<recipe>/skill_tests/model-flops-utilization/` 下补 recipe-local 测试：

- `test_spec.yaml`：声明这个 skill 在该 recipe 上要求哪些测试层级。
- `test_structure.py`：至少验证 recipe import、registry 接线、config schema 校验、
  required slots，以及 logger/checkpoint hooks；还必须验证注入的方法存在，
  且 MFU 的日志 key 已接线。
- `test_runtime.py`：至少成功构建 dataset、collator、model、optimizer、
  scheduler 和 engine，且不直接启动训练；还必须验证 engine 侧 MFU 计算
  能够在该 recipe 自己的运行时输入上被触发。
- `test_smoke.py`：覆盖 1 个真实、recipe-owned 的 single step：forward、loss、
  backward、optimizer step、logger write，以及 checkpoint noop 或临时保存；
  还必须验证同一步里能记录 `perf/mfu`，且不会破坏训练路径。
- `test_smoke.py` 必须走该 skill 的完整真实能力路径：真实 engine、真实 recipe
  入口，以及真实 MFU / logger / checkpoint 接线；禁止用 monkeypatch、fake MFU
  calculator、fake timer、fake logger 或类似测试桩把要验证的能力短路掉。
- 如果该 recipe 的 full-capability single-step 只能在 GPU 或分布式环境下成立，
  就把 smoke test 写成真实 launcher 测试，并在 `test_spec.yaml` 里把
  `gpu_preferred` 设为 `true`；不要为了在更弱环境里跑通而退化成 fake 逻辑。

这些测试必须走用户自己的 recipe / model 真实入口，不要换成绕开真实训练流的 toy model。

当你在用户 recipe 上执行这个 skill 时，应默认自动补齐这些测试，不要等用户自己提出。
验证必须且只能交给全新的 subagent，并使用 `fork_context=false`。禁止主 agent
在本地终端、后台终端会话或其他任何非 subagent shell fallback 中直接运行这些
`python -m tests.test_skills` 命令。先启动一个 subagent 运行
`python -m tests.test_skills --recipe <recipe> --skill model-flops-utilization --layer structure`，
只有它通过后，主 agent 才再启动新的 subagent 运行 `--layer runtime`；只有
runtime 通过后，主 agent 才再启动新的 subagent 运行 `--layer smoke`。最后由
主 agent 统一汇总三个层级的结果。如果 `test_smoke.py` 因 GPU、分布式启动条件
或执行权限限制而无法运行，主 agent 直接把准确的 `python -m tests.test_skills`
命令以及所需附加启动命令返回给用户。

## Output

- 说明 `calculate_model_flops(...)` 注入在了哪里。
- 说明 engine 侧 MFU 被加到了哪里或接入到了哪个已有位置。
- 说明峰值算力查表用了哪张 GPU、哪种精度。
- 说明当前按单卡还是多卡假设计算。
- 说明 `perf/mfu` 会出现在什么日志位置。
- 如果当前没有 GPU 且本地也没有启动说明，明确写出仍需用户提供 GPU 运行方式。

## Read On Demand

- 需要查询常见 GPU 在不同精度下的峰值算力时，读取 `references/hardware_peak_flops.csv`。
- 需要参考一个完整的 MFU 接线样例时，读取 `references/recipes/vit_classification_addon/` 下的 config、engine 和 model 文件。
