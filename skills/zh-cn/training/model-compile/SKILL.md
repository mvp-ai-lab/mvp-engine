---
name: model-compile
description: 为 recipe 接入或调整 model.compile，判断 compile 放在 parallelize 前后、哪些模块需要一起编译、如何暴露配置并做正确性与性能验证。适用于新模型启用 compile、已有 recipe 调整 compile 顺序、排查 compile 回归。
---

# Model Compile

## 目标

给 `recipes/<recipe>/` 下的训练 recipe 增加或调整 `model.compile` 支持，并保持：
- 默认关闭、配置显式开启。
- compile 作用在训练实际调用的模块上。
- compile 一定要放在 parallelize_model 之前。

## 本仓库约定

- 配置键放在 `optim` 下：
  - `optim.compile`
  - `optim.compile_backend`
  - `optim.compile_mode`
- compile 逻辑通常放在 `prepare_model()`。
- 不要编译 optimizer、scheduler、dataloader。

## 工作流

### 1. 先收集上下文

- 找到 recipe 的 `prepare_model()`。确认基础的模型构建已经完成。
- 搜索仓库内相近 recipe 作为先例：

```bash
rg -n "torch\\.compile|optim\\.compile|compile_backend|compile_mode" recipes
```

如果需要具体先例，按需读 `references/recipe-patterns.md`。

### 2. 决定 compile 范围

- 只 compile 训练热路径上的模块。
- 确认是否还有 teacher、EMA、辅助 head、蒸馏分支等独立 `forward()` 路径。如果有，询问是否都需要 compile。

### 3. 决定 compile 顺序

默认优先：
- 先 `model.compile(...)`，再 `parallelize_model(...)`。

硬性要求：
- 如果不用默认顺序，必须在代码注释或提交说明里写清原因。

### 4. 实现配置与代码

推荐模式：

```python
if bool(OmegaConf.select(self.config, "optim.compile", default=False)):
    model = model.compile(
        backend=OmegaConf.select(self.config, "optim.compile_backend", default="inductor"),
        mode=OmegaConf.select(self.config, "optim.compile_mode", default="default"),
    )
```

规则：
- `optim.compile` 必须有 `False` 默认值。
- `backend` 和 `mode` 用 `OmegaConf.select(..., default=...)` 读取。
- teacher/EMA 等额外模块分别 compile，不要隐式绑在主模型逻辑里。
- 不要为了 compile 改写 checkpoint 格式、参数命名或模型对外接口。

### 5. 验证

至少完成：
- config 验证

如果有 GPU 可以用，询问用户是否做如下测试：
- 单卡或单进程 `forward/backward` 冒烟。
- 对比 compile 开/关的 loss 与日志是否正常，不要求逐 bit 一致，但要无明显发散。

建议记录：
- 首步编译耗时。
- 稳态吞吐变化。
- 显存变化。

## 验收清单

- [ ] `optim.compile`、`optim.compile_backend`、`optim.compile_mode` 已接入 config。
- [ ] compile 目标模块与训练真实热路径一致。
- [ ] compile 顺序有明确依据；若是例外顺序，已注明原因。
- [ ] 额外模块、分支已逐个评估是否需要 compile。
