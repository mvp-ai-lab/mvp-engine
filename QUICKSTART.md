# Quickstart

这个仓库最重要的概念不是某个 API，而是一个很简单的对应关系：

> 你要训练什么 task，就先找最接近的 recipe；如果没有，就创建自己的 recipe。

`recipe` 可以理解成“一个训练任务的完整最小样板”。它通常包含这件事自己的配置、数据读取、模型构建和训练 step。

`mvp_engine/` 是通用训练底座，负责启动、训练循环、日志、checkpoint、分布式等公共能力。

`skills/` 是常用训练功能的操作指南。你可以让 agent 参考这些指南，把 gradient checkpointing、tensor parallel、model compile 等能力插入你的 recipes。

## 先从 task 找 recipe

| 你要做的任务 | 先看哪个 recipe | 为什么 |
| --- | --- | --- |
| 用 ViT 做图像分类 | `recipes/vit_classification/` | 最适合作为第一个模板，默认 fake data，可直接跑通 |
| 训练一个最小语言模型 | `recipes/magic_transformer/` | 使用 fake token data 做 next-token prediction，适合看纯文本 LM recipe 结构 |
| 微调 VLM | `recipes/minimal_vlm/` | 展示 JSONL + 图片路径的多模态 SFT 数据格式和训练方式 |
| 参考复杂多阶段 VLM 训练 | `recipes/openbee/` | 展示 alignment / pretrain / SFT 多阶段配置，使用 `stage1.yaml`、`stage2.yaml`、`stage3.yaml` |

第一次使用时，建议先从 `recipes/vit_classification/` 开始。它足够小，能直接展示 mvp-engine 的工作方式：recipe 负责定义“怎么训练这个任务”，engine 负责复用通用训练流程。

## 跑通第一个例子

先准备环境：

```bash
uv venv --python=3.12
source .venv/bin/activate
uv sync
```

然后运行 ViT 分类 recipe：

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/vit_classification/configs/train.yaml
```

这个 recipe 默认使用 fake data，所以你不需要先准备 ImageNet 就能看见训练流程跑起来。

这条命令背后发生了几件事：

- `--config` 指向某个 recipe 的配置文件。
- `mvp_engine.launch` 根据 config 路径自动 import 这个 recipe 下面的 Python 文件。
- `recipes/vit_classification/configs/train.yaml` 里的 `engine: ViTClassificationEngine` 决定使用哪个训练 Engine。
- `ViTClassificationEngine` 把 dataset、model、optimizer、scheduler 接到 `mvp_engine` 的通用训练循环里。

## 创建自己的 Recipes

### ➡️ Step 1: 先改 config - 训练参数是什么

创建自己的 recipe 时，通常先从 recipe 的 `train.yaml` 开始改，而不是去改 `mvp_engine/` 里的代码。

框架默认配置在：

```text
mvp_engine/config/default.yaml
```

这里放的是通用训练默认值，比如：

- `project`: 输出目录和项目名。
- `log`: 终端、文件、wandb 等日志后端。
- `parallel`: 分布式 device mesh。
- `optim`: 学习率、weight decay、混合精度、梯度累积。
- `loop`: 训练步数。
- `checkpoint`: 保存间隔和保留数量。

你的 recipe 配置通常在：

```text
recipes/<your_recipe>/configs/train.yaml
```

这个文件负责两件事：

- 指定这个任务用哪个 engine，例如 ViT 分类 recipe 里是 `engine: ViTClassificationEngine`。
- 覆盖或新增这个任务需要的字段，例如图像分类任务里的 `data` 和 `model`。

---

#### **数据方式**

mvp-engine 提供或当前 recipes 展示的数据入口：

| 数据方式 | 参考位置 | 适合场景 |
| --- | --- | --- |
| ImageFolder | `recipes/vit_classification/` | 图像分类 |
| fake token dataset | `recipes/magic_transformer/` | 最小语言模型 / next-token prediction smoke run |
| JSONL + 图片路径 | `recipes/minimal_vlm/` | VLM SFT |
| parquet / staged VLM data | `recipes/openbee/configs/stage*.yaml` | 多阶段 VLM 训练 |
| WebDataset tar shards | `mvp_engine/dataset/webdataset.py` | 大规模 tar shard 数据 |


---

#### **分布式策略**

分布式策略主要看 `parallel.mesh`：

```yaml
parallel:
  mesh:
    replicate: -1
    shard: 1
    tensor: 1
```

三个维度的含义是：

- `replicate`: 数据并行复制维度。
- `shard`: FSDP2 分片维度。
- `tensor`: tensor parallel 维度。

常见理解方式：

- `-1` 表示这个维度根据 world size 自动推断。
- `shard: 1, tensor: 1` 是 DDP 风格。
- `shard > 1` 会走 FSDP2 相关路径。
- `tensor > 1` 会启用 tensor parallel 相关路径。

---

#### **模型权重**

mvp-engine 不限制权重来源，具体怎么加载由 recipe 的 `model/` 和 `engine.prepare_model()` 决定。常见方式有两种：

- 从本地 checkpoint 或本地模型目录加载。
- 从 Hugging Face 等模型仓库加载预训练权重。

也就是说，`mvp_engine/` 不会强行规定权重格式；你只需要在 recipe 里把模型构建和权重加载逻辑接好。

---

#### **其他参数**

大多数训练实验还会先改这些字段：

```yaml
optim:
  lr: 0.0001
  weight_decay: 0.05
  mixed_precision: "bf16"
  gradient_accumulation_steps: 1

loop:
  total_steps: 1000

checkpoint:
  interval: 500

log:
  backends: ["terminal", "file"]
```

这些字段分别控制学习率、weight decay、混合精度、梯度累积、总训练步数、checkpoint 保存间隔和日志输出位置。



---

#### **以 ViT classification 为例**

`recipes/vit_classification/configs/train.yaml` 展示了一个图像分类 recipe 通常怎么选择 data、distributed 和 model。
下面是 `train.yaml` 中相关的字段节选：

```yaml
data:
  use_fake_data: true
  train_path: "./data/imagenet/train"
  val_path: "./data/imagenet/val"
  fake_train_size: 1024
  fake_val_size: 256
  num_classes: 1000
  image_size: 224
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  batch_size: 64
  num_workers: 4

model:
  pretrained_model_name_or_path: "google/vit-base-patch16-224"
  load_pretrained_weights: false
  num_classes: 1000
  image_size: 224
  hidden_dropout_prob: 0.0
  attention_dropout_prob: 0.0

parallel:
  mesh:
    replicate: -1
    shard: 8
    tensor: 1
  backend_kwargs:
    fsdp2:
      reshard_after_forward: true
      offload_policy: false
      mp_policy:
        param_dtype: bfloat16
        reduce_dtype: float32
        output_dtype: bfloat16
      high_precision_modules: ["LayerNorm", "RMSNorm"]
```

这段配置里的选择可以这样读：

- data: 默认 `use_fake_data: true`，所以可以不准备真实数据就先跑通；如果切到真实数据，使用 `ImageFolder` 格式，通过 `train_path` 和 `val_path` 指向训练/验证目录。
- model: 使用 `google/vit-base-patch16-224` 对应的 ViT 结构；默认 `load_pretrained_weights: false`，保持 smoke run 离线友好。
- distributed: `replicate: -1` 表示数据并行复制维度由 world size 自动推断，`shard: 8` 表示启用 FSDP2 shard 维度，`tensor: 1` 表示不启用 tensor parallel。

### ➡️ Step 2: 再改 Engine - 任务怎么接入训练循环

新任务通常要改动或实现这些 hook：

| Hook | 你在这里做什么 |
| --- | --- |
| `prepare_dataloader` | 构建训练和评估 dataloader，决定数据怎么采样、batch 怎么组织 |
| `prepare_model` | 构建模型，并把模型交给 `parallelize_model` 做 DDP、FSDP2 或 TP 包装 |
| `prepare_optimizer` | 选择 optimizer，并决定哪些参数参与训练 |
| `prepare_scheduler` | 构建学习率 scheduler |
| `train_pre_step` | 把 dataloader 产出的 batch 移到当前 device，并整理成模型需要的输入格式 |
| `train_one_step` | 做 forward，计算 loss，返回日志指标 |

🔔: `mvp_engine` 的通用训练循环要求 `train_one_step` 至少返回：

```python
{"loss": loss_tensor, "logs": {"train/loss": loss_value}}
```

其中：

- `loss` 必须是可以反向传播的 tensor。
- `logs` 是要记录的标量指标，比如 loss、accuracy、learning rate 以外的任务指标。

---

#### **以 `ViTClassificationEngine` 为例**

`ViTClassificationEngine` 对应的任务是“图片分类”，所以它只需要把图片和标签接到通用训练循环里：

- `prepare_dataloader` 调用 `build_dataset(self.config, workflow)`。这个 builder 会根据 `data.use_fake_data` 选择 `FakeData` 或 `ImageFolder`；训练时使用 `InfiniteDistributedSampler`，评估时使用 `DistributedSampler`。
- `prepare_model` 调用 `build_vit_model(self.config.model)` 构建 ViT 分类模型，再调用 `parallelize_model` 按 `parallel.mesh` 和 `backend_kwargs` 接入 DDP、FSDP2 或 TP。
- `prepare_optimizer` 使用 `AdamW`，学习率和 weight decay 来自 `config.optim.lr` 和 `config.optim.weight_decay`。
- `prepare_scheduler` 使用 warmup + cosine schedule，warmup 比例来自 `config.optim.warmup_ratio`，总步数来自 `loop.total_steps`。
- `train_pre_step` 接收 dataloader 产出的 `(pixel_values, labels)`，移动到当前 device，并整理成 `{"pixel_values": ..., "labels": ...}`。
- `train_one_step` 调用 `self.model(pixel_values=..., labels=...)`，返回 `outputs.loss`，同时把 `train/loss` 和 `train/acc1` 写进 logs。


### ✅ 总结: 最少要 build 什么

复制一个最接近的 recipe，或者从零建立一个自己的 recipe 到：

```text
recipes/<your_recipe>/
```

然后最少实现四块：

- `configs/schema.py`: 定义这个任务需要的配置字段。
- `dataset/`: 读取你的数据，并整理出训练 batch。
- `model/`: 构建你的 `torch.nn.Module`。
- `engine/`: 把 dataloader、model、optimizer、scheduler 接起来，并定义训练 step。

😊 **GOOD NEWS**: 新任务不需要重写完整训练循环。你只需要在 recipe 里告诉框架：

- 数据怎么来。
- 模型怎么建。
- 一个 batch 怎么前向并算 loss。

剩下的 backward、gradient accumulation、mixed precision、scheduler step、日志和 checkpoint，都由 `mvp_engine` 处理。

## 🔧 可用的 SKILLS

当前 skills 使用 `skills/<category>/<skill-name>/SKILL.md` 的目录结构。

| Category/Skill | Description |
| --- | --- |
| `training/gradient-checkpointing` | 为 recipe 接入 gradient checkpointing，并补齐配置、engine 接线和验证。 |
| `training/model-compile` | 为 recipe 增加或调整 `torch.compile`，选择合适 compile 范围并验证效果。 |
| `parallel/tensor-parallel` | 为模型生成 recipe-local tensor parallel plan，并调整 mesh 配置。 |
| `parallel/fsdp2-prefetching` | 为复杂模型生成 FSDP2 prefetch callable，改善通信与计算重叠。 |
| `model/model-migration` | 将外部模型迁入 `recipes/<recipe>/model/`，保持 checkpoint 兼容。 |
| `model/model-flops-utilization` | 接入模型 FLOPs、硬件峰值算力和 MFU 日志。 |
| `recipe/new-recipe-template` | 在 `recipes/` 下创建标准 recipe 脚手架。 |
| `experiment/analysis` | 分析 `outputs/<run_id>/` 产物并生成实验报告。 |
| `git/pr-gate` | push 或开 PR 前执行质量门禁、lint/test 和风险总结。 |
| `git/pr-feedback` | 处理 PR reviewer 反馈，完成定向修复和回复草稿。 |
| `git/pr-skill-review` | 专门评审 `skills/` 改动。 |
| `git/recipe-merge-repair` | 将上游更新安全合入 recipe 分支并完成适配验证。 |
| `skills/create-a-skill` | 创建或规范化新的 skill。 |
