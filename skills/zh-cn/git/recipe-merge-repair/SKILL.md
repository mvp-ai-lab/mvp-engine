---
name: recipe-merge-repair
description: 比较当前开发分支与上游分支的异同，识别会影响目标 recipe 的共享契约变化，在当前分支上完成适配、冲突解决、验证，并将上游分支成功 merge 进来。适用于 main 等基线分支有重大更新，而当前 recipe 开发分支需要同步这些能力的场景。
---

# recipe-merge-repair

> 中文版：`skills/zh-cn/git/recipe-merge-repair/SKILL.md`
> English version: `skills/en/git/recipe-merge-repair/SKILL.md`

## Goal

- 在真正 merge 之前，先搞清楚当前开发分支与上游分支的差异。
- 把 branch diff 转成 recipe breakage / adaptation 清单，而不是边 merge 边猜。
- 在当前开发分支中完成 recipe-local 适配、冲突解决和验证。
- 最终把上游分支安全地 merge 到当前开发分支。
- 优先把修复留在 `recipes/<recipe>/` 内，除非已经明确是共享引擎本身的通用 bug。

## Required Inputs

- 当前开发分支名，以及要引入特性的上游分支名，例如 `main`。
- 目标 recipe 路径、配置路径，或 `recipes/` 下的 engine/model 文件。
- 如果已知，再提供 merge base、commit 区间，或重点 commits/files 列表。
- 修复后的验证目标：
  - 只做 config/schema 验证，
  - 做 model/engine smoke test，
  - 或做完整训练启动验证。
- 当前工作树是否干净；如果不干净，要先判断未提交改动是否与本次 merge 相关。
- 相关运行要求：
  - `source ~/.bashrc`
  - `source .venv/bin/activate`
  - GPU 验证前先使用本地集群命令或 alias

## Workflow

### 1. 先建立 merge 上下文和保护边界

- 不要一上来就 `git merge`。先确认：
  - 当前分支是谁
  - 上游分支是谁
  - 两者的 merge base 是谁
  - 当前工作树是否包含未提交改动
- 优先使用：
  - `git branch --show-current`
  - `git status --short`
  - `git merge-base <current_branch> <upstream_branch>`
- 如果工作树不干净，不要让 merge 覆盖不明确的本地修改。
- 先把当前分支要保留的 recipe 目标、配置入口、验证目标记清楚，再进入 diff 分析。

### 2. 在 merge 前比较两个分支的异同

- 先分清三类变化：
  - 上游分支新增了什么
  - 当前开发分支本地新增了什么
  - 两边在哪些文件或契约上重叠
- 优先使用：
  - `git log --left-right --graph --oneline <merge_base>...<upstream_branch>`
  - `git log --left-right --graph --oneline <merge_base>...<current_branch>`
  - `git diff --name-status <merge_base>..<upstream_branch>`
  - `git diff --name-status <merge_base>..<current_branch>`
  - `git diff --name-status <current_branch>...<upstream_branch>`
  - `git diff <merge_base>..<upstream_branch> -- mvp_engine/ recipes/<target_recipe>/`
  - `git diff <merge_base>..<current_branch> -- recipes/<target_recipe>/`
- 如果当前分支也改过共享代码，必要时再看：
  - `git range-diff <merge_base>..<upstream_branch> <merge_base>..<current_branch>`
- 按契约类型分组：
  - config schema / layout 变化
  - engine 生命周期或方法签名变化
  - 分布式或 checkpoint 运行时变化
  - model wrapper 或 registry 变化
  - dataset / dataloader 接口变化

### 3. 把 branch diff 映射到 recipe 表面

- 读取目标 recipe 的：
  - `engine/*.py`
  - `model/**/*.py`
  - `configs/*.yaml`
  - 如果存在，再读 recipe-local `configs/schema.py`
- 找出 recipe 对上游新契约和当前分支本地改动的直接依赖。
- 不要发现一个报错就立刻停下。先把 breakage list 和 merge hotspot list 建完整，再开始改。

本仓库里的常见模式：

- 核心 config 系统重构了，但 recipe 还在依赖原始 `DictConfig` 风格，没有 recipe-local `ConfigClass`
- 共享并行接口变了：
  - mesh key 改名
  - `backend_kwargs` 变成按 backend 分层
  - `parallelize_model(...)` 签名变化
  - checkpoint helper 改为从 `DeviceMesh` 自动推断 backend
- 引入或重构了 TP 运行时，但 recipe model 没有 `TP_MODULE_CONFIG`
- 配置字段从旧的嵌套位置搬到了新的顶层位置，而 recipe YAML 仍然沿用旧写法

### 4. 先做 merge/adaptation 计划，再真正解决冲突

- 对每个 hotspot，先决定策略：
  - 直接接受上游版本
  - 保留当前分支 recipe-local 设计
  - 手工组合两边改动
  - 识别为共享层通用 bug，改 `mvp_engine/`
- 典型判断：
  - 如果上游改的是共享契约，而当前分支改的是 recipe 接线，通常要保留两边并在 recipe 层做适配。
  - 如果当前分支只是复制了旧共享逻辑，上游已经提供了新共享实现，不要把旧逻辑硬合回来；要迁移到新契约。
  - 不要只按冲突块逐行拼接；先判断运行时契约到底该长什么样。

### 5. 在正确层次修 recipe 并解决 merge 冲突

- 优先做 recipe-local 修复：
  - 新增或更新 `recipes/<recipe>/configs/schema.py`
  - 在 recipe engine 上设置 `ConfigClass`
  - 把 recipe YAML 迁移到当前共享 schema
  - 更新 recipe-local engine/model 代码以匹配当前共享运行时契约
- 只有在 merged code 本身会影响所有 recipe 时，才改 `mvp_engine/`。
- 不要过度抽象。修复代码保持直接、局部、可读。
- 当分析已经充分后，再执行真正的 merge，优先使用：
  - `git merge --no-commit --no-ff <upstream_branch>`
- 先根据前面的 hotspot map 解决冲突，再补必要的 recipe-local 适配。
- merge 完成前，不要因为“冲突消失了”就当成问题已解决；必须继续做契约级检查。

### 6. 显式处理 config-schema 断层

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

### 7. 显式处理 distributed-runtime 断层

- 更新 recipe 对共享 helper 的调用，让它们符合当前签名。
- 以本仓库当前代码为准：
  - `parallelize_model(...)` 只接收 `model`、`device_mesh`、`backend_kwargs`
  - checkpoint helper 第一个参数是 `mesh`，DDP/FSDP2 由 helper 自行推断
- 如果 recipe 之前依赖 `self.parallel_backend`，改成从 mesh 推断，或直接依赖新的 helper 行为。
- 如果任何 config 启用了 TP，确认模型类定义了合法的 `TP_MODULE_CONFIG`；只有当 forward 仍然依赖全局维度缓存时才加 `TP_MODULE_POSTPROCESSORS`。

### 8. 按目标路径验证 merge 结果

- 用能证明 merge breakage 已修好的最小验证层级：
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

### 9. 完成 merge 并做收尾检查

- merge 冲突解决并验证通过后，再完成 merge commit。
- 至少补做：
  - `git diff --check`
  - `git status --short`
- 确认最终结果是：
  - 上游特性已经进入当前开发分支
  - 当前 recipe 的本地开发目标没有被上游改动抹掉
  - 验证路径与 merge 结果一致，而不是只验证了 pre-merge 状态

## Validation

- 确认先做了 branch comparison，再开始 merge 和修复。
- 确认区分清楚了：
  - merge base 之后的上游变化
  - merge base 之后的当前分支变化
  - 两边重叠的热点文件与契约
- 确认 recipe 的代码和 YAML 都已经对齐当前共享契约。
- 对每一类 breakage 都执行对应的 targeted validation。
- 确认验证是在 merge 后的工作树上执行的。
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
  - 当前分支、上游分支、merge base
  - 检查过哪些 branch-only files / commits
  - 哪些文件或契约是 merge hotspots
  - 在目标 recipe 中发现了哪些 breakages
  - 应用了哪些 recipe-local 修复
  - 如何解决关键 merge conflicts
  - 跑了哪些验证命令、结果如何
  - 还剩哪些风险或未验证路径

## Read On Demand

- 当场景是“上游 shared config / distributed refactor 很大，当前 recipe 分支需要把它 merge 进来”时，读取 `references/tomatovit-parallel-refactor-example.md`。
- 当你需要一个“当前仓库里健康 recipe 应该怎样验证”的基线时，读取 `references/vit-classification-baseline-example.md`。
