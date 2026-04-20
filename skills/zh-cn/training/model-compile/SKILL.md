---
name: model-compile
description: 为 recipe 增加或调整 model.compile，判断 compile 范围与放置位置、在 model 下暴露配置，并验证正确性与性能。
---

# Model Compile

## Goal

- 为 `recipes/<recipe>/` 下的训练 recipe 增加或调整 `model.compile` 支持。
- 让 compile 默认关闭、只通过配置显式开启。
- 让 compile 作用在真实训练热路径的模块上，并且除非有明确理由，否则保持在 `parallelize_model` 之前。

## Required Inputs

- 目标 recipe 路径和其 `prepare_model()` 实现。
- 真实训练热路径上的候选模块。
- recipe 是否还有 teacher、EMA、辅助 head 或其他独立分支。
- 目标 recipe 的 config 或 schema 文件。
- 如果要做正确性或性能验证，是否有可用 GPU。

## Workflow

### 1. 先收集上下文

- 找到 recipe 的 `prepare_model()`，确认基础模型构建已经完成。
- 如果 `references/` 下有匹配目标 recipe 的参考实现，先读参考文件。
- 搜索仓库内其他 compile 先例：

```bash
rg -n "torch\.compile|model\.compile|compile_backend|compile_mode" recipes
```

### 2. 决定 compile 范围

- 只 compile 训练热路径上的模块。
- 如果顶层 `forward()` 混有大量 Python 预处理、token 构造、位置编码准备或其他 recipe 胶水逻辑，默认不要直接编整个模型。
- 当 recipe 还包含 teacher、EMA、辅助 head 或蒸馏分支时，分别评估这些分支，而不是把它们隐藏在主模型决定里。
- 优先选择一个 compile-friendly 的核心目标，而不是把 compile 切成很多零碎的小子模块。

### 3. 决定 compile 顺序

- 默认顺序是：
  - 先调用 `model.compile(...)`
  - 再调用 `parallelize_model(...)`
- 如果某个 recipe 需要其他顺序，必须在代码注释或变更说明里写清原因。

### 4. 实现配置与代码

- 把 compile 配置放在 `model` 下：
  - `model.compile`
  - `model.compile_backend`
  - `model.compile_mode`
- 通过 recipe 的 schema 或 `ConfigClass` 暴露这些字段。
- 在 `prepare_model()` 中按如下模式接线：

```python
if self.config.model.compile:
    model.compile(
        backend=self.config.model.compile_backend,
        mode=self.config.model.compile_mode,
    )
```

- `model.compile` 默认值必须是 `False`。
- 不要为了迁就 compile 而改 checkpoint 格式、参数命名或对外接口。

### 5. 验证正确性与性能

- 至少验证 config 解析和 compile 接线本身。
- 如果 GPU 可用，询问用户是否执行：
  - 单进程或单卡的 forward/backward 冒烟测试
  - compile 开关前后的 loss 和日志对比
- 在可行时记录首步编译耗时、是否进入稳态、吞吐变化和显存变化。

## Validation

- `model.compile`、`model.compile_backend` 和 `model.compile_mode` 已接入配置。
- 被 compile 的目标与真实训练热路径一致。
- compile 没有在缺乏证据的情况下被切碎成很多小子模块。
- compile 顺序要么是默认顺序，要么有明确的例外说明。
- teacher、EMA 等额外分支都被单独评估过。

在 `recipes/<recipe>/skill_tests/model-compile/` 下补 recipe-local 测试：

- `test_spec.yaml`：声明这个 skill 在该 recipe 上要求哪些测试层级。
- `test_structure.py`：至少验证 recipe import、registry 接线、config schema 校验、
  required slots，以及 logger/checkpoint hooks；还必须验证配置字段存在，且
  compile 接线位置符合预期。
- `test_runtime.py`：至少成功构建 dataset、collator、model、optimizer、
  scheduler 和 engine，且不直接启动训练；还必须验证 `torch.compile`
  只作用在预期的训练热路径模块上。
- `test_smoke.py`：覆盖 1 个真实、recipe-owned 的 single step：forward、loss、
  backward、optimizer step、logger write，以及 checkpoint noop 或临时保存；
  还必须验证 compile 开/关两种路径都能走通该 recipe 自己的训练路径。
- `test_smoke.py` 必须走该 skill 的完整真实能力路径：真实 engine、真实 recipe
  入口，以及真实 `torch.compile` / logger / checkpoint 接线；禁止用 monkeypatch、
  fake compile wrapper、fake training step 或类似测试桩把要验证的能力短路掉。
- 如果该 recipe 的 full-capability single-step 只能在 GPU 或分布式环境下成立，
  就把 smoke test 写成真实 launcher 测试，并在 `test_spec.yaml` 里把
  `gpu_preferred` 设为 `true`；不要为了在更弱环境里跑通而退化成 fake 逻辑。

如果有 GPU 可以用，询问用户是否做如下测试：
- 单卡或单进程 `forward/backward` 冒烟。
- 对比 compile 开/关的 loss 与日志是否正常，不要求逐 bit 一致，但要无明显发散。

这些测试必须走用户自己的 recipe / model 真实入口，只能缩到 recipe 自己的最小配置或最小 batch，
不要替换成无关的 tiny model。

当你在用户 recipe 上执行这个 skill 时，应默认自动补齐这些测试，不要等用户再单独要求。
验证应交给全新的 subagent 且 `fork_context=false`：先启动一个 subagent 运行
`python -m tests.test_skills --recipe <recipe> --skill model-compile --layer structure`，
只有它通过后，主 agent 才再启动新的 subagent 运行 `--layer runtime`；只有
runtime 通过后，主 agent 才再启动新的 subagent 运行 `--layer smoke`。最后由
主 agent 统一汇总三个层级的结果。如果 `test_smoke.py` 因 GPU、分布式启动条件
或执行权限限制而无法运行，主 agent 直接把准确的 `python -m tests.test_skills`
命令以及所需附加启动命令返回给用户。

## Output

- 说明更新了哪些 model、engine 和 config 文件。
- 说明最终被 compile 的模块或可调用对象是什么。
- 说明采用的 compile 顺序，以及是否偏离默认顺序。
- 总结已执行的正确性或性能验证，以及仍未验证的部分。

## Read On Demand

- 需要当前 compile 接线参考实现时，读取 `references/vit_classification/configs/train.yaml` 和 `references/vit_classification/engine/vit_classification_engine.py`。
