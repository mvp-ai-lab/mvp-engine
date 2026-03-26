---
name: recipe-merge-repair
description: 检查最近合入的共享代码是否破坏当前 recipe，并完成 recipe-local 代码或配置修复与验证。适用于 mvp_engine/ 或其他共享模块更新后，怀疑 recipes/ 下某个 recipe 已失效的场景。
---

# recipe-merge-repair

> 中文版：`skills/zh-cn/debug/recipe-merge-repair/SKILL.md`
> English version: `skills/en/debug/recipe-merge-repair/SKILL.md`

## Goal

- 判断最近合入的共享代码是否影响当前正在处理的 recipe。
- 如果共享改动把 recipe 弄坏了，在停止前把 recipe 修好。
- 优先把修复留在 `recipes/<recipe>/` 内，除非已经明确是共享引擎本身的通用 bug。

## Required Inputs

- 目标 recipe 路径、配置路径，或 `recipes/` 下的 engine/model 文件。
- 合入改动的范围：
  - 基线分支或 commit 区间，或
  - 已知的 merged commits/files 列表。
- 修复后的验证目标：
  - 只做 config/schema 验证，
  - 做 model/engine smoke test，
  - 或做完整训练启动验证。
- 相关运行要求：
  - `source ~/.bashrc`
  - `source .venv/bin/activate`
  - GPU 验证前先使用本地集群命令或 alias

## Workflow

### 1. 先建立 merged-change 地图

- 在改 recipe 之前，先看共享代码到底改了什么。
- 优先使用：
  - `git log --oneline <base>..HEAD`
  - `git diff --name-status <base>..HEAD`
  - `git diff <base>..HEAD -- mvp_engine/ recipes/<target_recipe>/`
- 按契约类型分组：
  - config schema / layout 变化
  - engine 生命周期或方法签名变化
  - 分布式或 checkpoint 运行时变化
  - model wrapper 或 registry 变化
  - dataset / dataloader 接口变化

### 2. 把共享改动映射到 recipe 表面

- 读取目标 recipe 的：
  - `engine/*.py`
  - `model/**/*.py`
  - `configs/*.yaml`
  - 如果存在，再读 recipe-local `configs/schema.py`
- 找出 recipe 对新契约的直接依赖。
- 不要发现一个报错就立刻停下。先把 breakage list 建完整，再开始改。

本仓库里的常见模式：

- 核心 config 系统重构了，但 recipe 还在依赖原始 `DictConfig` 风格，没有 recipe-local `ConfigClass`
- 共享并行接口变了：
  - mesh key 改名
  - `backend_kwargs` 变成按 backend 分层
  - `parallelize_model(...)` 签名变化
  - checkpoint helper 改为从 `DeviceMesh` 自动推断 backend
- 引入或重构了 TP 运行时，但 recipe model 没有 `TP_MODULE_CONFIG`
- 配置字段从旧的嵌套位置搬到了新的顶层位置，而 recipe YAML 仍然沿用旧写法

### 3. 在正确层次修 recipe

- 优先做 recipe-local 修复：
  - 新增或更新 `recipes/<recipe>/configs/schema.py`
  - 在 recipe engine 上设置 `ConfigClass`
  - 把 recipe YAML 迁移到当前共享 schema
  - 更新 recipe-local engine/model 代码以匹配当前共享运行时契约
- 只有在 merged code 本身会影响所有 recipe 时，才改 `mvp_engine/`。
- 不要过度抽象。修复代码保持直接、局部、可读。

### 4. 显式处理 config-schema 断层

- 如果 recipe engine 继承自 `Engine`，并且会访问 `BaseEngineConfig` 之外的字段，就必须补 recipe-local schema。
- schema 里要覆盖 engine/model 实际读取到的所有 recipe 字段。
- 特别检查 Pydantic 校验导致的静默字段丢失：
  - `data.*`
  - `model.*`
  - recipe 专属的 `optim.*`
  - 曾经放在旧嵌套块里的 checkpoint 配置
- 对模板型 recipe，还要检查 schema 中允许的调参字段是否真的暴露在 YAML 里。
  - 如果字段存在于 recipe schema，但没有写进当前配置文件，那么在 struct mode 下，普通 Hydra override 可能会直接失败。
  - 对 smoke test 常用的调参项，优先在 YAML 中写出显式默认值，而不是逼使用者总写 `+foo.bar=...`。
- 如果 recipe 还在用旧 mesh key，例如 `dp_size`、`fsdp2_size`、`tp_size`，改成：
  - `parallel.mesh.replicate`
  - `parallel.mesh.shard`
  - `parallel.mesh.tensor`
- 如果 smoke run 使用 `WORLD_SIZE=1`，要确认修完后的 mesh 仍然能推导出合法尺寸。
  - 像 `replicate: -1, shard: 8` 这样的模板配置，在单卡验证时会推导出 `replicate=0`，还没进入 recipe 逻辑就会报错。
  - 对单卡 smoke test，优先临时改成 `replicate=1, shard=1, tensor=1`，除非该 recipe 明确依赖 sharding。
- 如果 `backend_kwargs` 已改成按 backend 分层，改成：
  - `parallel.backend_kwargs.fsdp2.*`
  - `parallel.backend_kwargs.ddp.*`

### 5. 显式处理 distributed-runtime 断层

- 更新 recipe 对共享 helper 的调用，让它们符合当前签名。
- 以本仓库当前代码为准：
  - `parallelize_model(...)` 只接收 `model`、`device_mesh`、`backend_kwargs`
  - checkpoint helper 第一个参数是 `mesh`，DDP/FSDP2 由 helper 自行推断
- 如果 recipe 之前依赖 `self.parallel_backend`，改成从 mesh 推断，或直接依赖新的 helper 行为。
- 如果任何 config 启用了 TP，确认模型类定义了合法的 `TP_MODULE_CONFIG`；只有当 forward 仍然依赖全局维度缓存时才加 `TP_MODULE_POSTPROCESSORS`。

### 6. 按目标路径验证修复

- 用能证明 breakage 已修好的最小验证层级：
  - schema/model import
  - engine 初始化
  - model parallelization smoke test
  - 训练启动
- 本仓库的 GPU 验证路径：
  - `source ~/.bashrc`
  - 使用本地集群命令或 alias 进入 GPU shell
  - `source .venv/bin/activate`
- 如果真实数据或预训练权重不可用，不要直接跳过验证；先构造最小的临时本地 fixture 做 smoke test。
- smoke test 优先使用临时 override，不要为了测试永久削弱 recipe 默认配置。

## Validation

- 确认先做了 merged-code inspection，再做修复。
- 确认 recipe 的代码和 YAML 都已经对齐当前共享契约。
- 对每一类 breakage 都执行对应的 targeted validation。
- 本仓库优先使用：

```bash
python -m compileall recipes/<recipe>
```

```bash
python - <<'PY'
# recipe-local config/schema smoke test
PY
```

```bash
source ~/.bashrc
<gpu-shell-command>
source .venv/bin/activate
python - <<'PY'
# GPU smoke test for model/engine startup
PY
```

- 如果补了 TP config，确认 plan key 与真实子模块名一致。
- 如果只跑了 smoke test，要明确说明哪些完整训练路径还没有被验证。

## Output

- 最终需要汇报：
  - 检查过哪些 merged files / commits
  - 在目标 recipe 中发现了哪些 breakages
  - 应用了哪些 recipe-local 修复
  - 跑了哪些验证命令、结果如何
  - 还剩哪些风险或未验证路径

## Read On Demand

- 当 breakage 看起来像“共享 config / distributed refactor 合入后，recipe 被静默弄坏”时，读取 `references/tomatovit-parallel-refactor-example.md`。
- 当你需要一个“当前仓库里健康 recipe 应该怎样验证”的基线时，读取 `references/vit-classification-baseline-example.md`。
