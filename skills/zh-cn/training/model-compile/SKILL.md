---
name: model-compile
description: 为 recipe 接入或调整 model.compile，判断 compile 范围与放置位置、在 optim 下暴露配置，并做正确性与性能验证。`references/` 下的文件是这个 skill 的参考实现。适用于新模型启用 compile、已有 recipe 调整 compile 顺序、排查 compile 回归。
---

# Model Compile

## 目标

给 `recipes/<recipe>/` 下的训练 recipe 增加或调整 `model.compile` 支持，并保持：
- 默认关闭、配置显式开启。
- compile 作用在训练实际调用的模块上。
- compile 一定要放在 parallelize_model 之前。

## 本仓库约定

- 配置键放在 `model` 下：
  - `model.compile`
  - `model.compile_backend`
  - `model.compile_mode`
- compile 逻辑通常放在 `prepare_model()`。
- 不要编译 optimizer、scheduler、dataloader。

## 工作流

### 1. 先收集上下文

- 找到 recipe 的 `prepare_model()`。确认基础的模型构建已经完成。
- 如果 `references/` 里有匹配目标 recipe 的参考实现，先读这些文件。它们是这个 skill 期望的 config 和 engine 接线范例。
- 搜索仓库内相近 recipe 作为补充先例：

```bash
rg -n "torch\\.compile|model\\.compile|compile_backend|compile_mode" recipes
```

对于当前 skill，`references/vit_classification/configs/train.yaml` 和
`references/vit_classification/engine/vit_classification_engine.py` 是当前参考实现。

### 2. 决定 compile 范围

- 只 compile 训练热路径上的模块。
- 确认是否还有 teacher、EMA、辅助 head、蒸馏分支等独立 `forward()` 路径。如果有，询问是否都需要 compile。
- 如果顶层 `forward()` 混有大量 Python 预处理、token 构造、位置编码准备、输出分支或其他 recipe 胶水逻辑，默认不要直接编整个模型。
- 这种情况下，先询问用户是否抽出一个 compile-friendly 的 core 模块/可调用对象，只覆盖稠密 tensor 热路径。
- 除非已经证明有效，否则不要把 compile 拆成很多很小的子模块；过碎的 compile 往往拿不到跨层融合，还会明显拉长首步编译时间。

### 3. 决定 compile 顺序

默认优先：
- 先 `model.compile(...)`，再 `parallelize_model(...)`。

硬性要求：
- 如果不用默认顺序，必须在代码注释或提交说明里写清原因。

### 4. 实现配置与代码

推荐模式：

```python
if self.config.model.compile:
    model.compile(
        backend=self.config.model.compile_backend,
        mode=self.config.model.compile_mode,
    )
```

规则：
- `model.compile` 必须有 `False` 默认值。
- 新配置系统下，recipe 要通过自己的 Pydantic `ConfigClass` 暴露 `model.compile*` 字段，并使用属性访问读取。
- teacher/EMA 等额外模块分别 compile，不要隐式绑在主模型逻辑里。
- 如果需要为了 compile 抽 recipe 专属的 encoder/core 子模块，优先编译一个较大的核心目标，而不是把几十个 block 分别 compile。
- 不要为了 compile 改写 checkpoint 格式、参数命名或模型对外接口。

### 5. 验证

至少完成：
- config 验证

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
如果因为 GPU 资源或执行权限限制而无法运行，直接把准确的 `tests/test_skills.py` 命令
以及所需附加启动命令返回给用户。

建议记录：
- 首步编译耗时。
- 是否真的进入了 step 2 / 稳态；如果 compile 只能勉强跑完 step 1，通常还不能算可用。
- 稳态吞吐变化。
- 显存变化。

## 验收清单

- [ ] `model.compile`、`model.compile_backend`、`model.compile_mode` 已接入 config。
- [ ] compile 目标模块与训练真实热路径一致。
- [ ] compile 目标没有被切得过碎；优先一个 compile-friendly core，而不是很多零散 compiled 子模块。
- [ ] compile 顺序有明确依据；若是例外顺序，已注明原因。
- [ ] 额外模块、分支已逐个评估是否需要 compile。
