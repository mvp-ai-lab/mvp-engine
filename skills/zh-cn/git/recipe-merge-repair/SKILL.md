---
name: recipe-merge-repair
description: 系统比较当前开发分支与上游分支，识别会影响目标 recipe 的共享变化与集成风险，将这些变化转成清晰的 breakage / adaptation 清单，并在当前分支上完成冲突解决、适配、验证与最终 merge。适用于基线分支发生显著更新，而某个在研 recipe 分支需要安全吸收这些更新的场景。
---

# recipe-merge-repair

> 中文版：`skills/zh-cn/git/recipe-merge-repair/SKILL.md`
> English version: `skills/en/git/recipe-merge-repair/SKILL.md`

## Goal

- 把上游分支安全地 merge 到当前开发分支，而不是停留在 diff 分析或文本级冲突处理。
- 解决 merge 过程中暴露出来的代码、配置、接口、运行路径和共享契约冲突，让目标 recipe 在合并后仍然可运行、可维护。
- 将当前 recipe 适配到上游分支引入的新实现和新行为，确保本地开发目标不会在 merge 后被破坏、回退或悄悄失效。
- 在 merge 完成后通过针对性验证确认修复真实成立，而不是只做到“冲突消失”。
- 优先把修复留在 `recipes/<recipe>/` 内，只有在问题明确属于共享层时才修改共享代码。

## Required Inputs

- 当前开发分支名，以及要引入更新的上游分支名，例如 `main`。
- 目标 recipe 的路径、入口配置、关键模块，或 `recipes/` 下相关的 engine/model/data 文件。
- 如果已知，再提供 merge base、commit 区间，或需要重点关注的 commits/files。
- 希望达到的验证范围与验证深度：
  - 只验证基础可用性或最小关键路径，
  - 验证主要运行路径的 smoke test，
  - 或做更完整的启动、训练、评估、推理或端到端验证。
- 当前工作树是否干净；如果不干净，要先判断未提交改动是否属于这次 merge 的范围。
- 如果验证依赖额外环境、数据、服务、权限或计算资源，需要提前说明这些前提。
- 如果用户同意，可以按用户指定的方式申请所需资源，并据此执行验证。

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
- 如果工作树不干净，不要让 merge 覆盖来源不明的本地修改。
- 在进入 diff 分析前，先明确这次 merge 要保住什么：目标 recipe、主要入口、关键能力、验证目标。

### 2. 在 merge 前比较两个分支的异同

- 先分清三类变化：
  - 上游分支新增或改动了什么
  - 当前开发分支本地新增或改动了什么
  - 两边在哪些文件、模块或契约上发生重叠
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
- 比较结果不要只按文件名整理，也要按能力面和契约面整理，例如：
  - 入口与配置
  - API、类、函数或 hook 的调用约定
  - 训练、评估、推理等生命周期
  - 数据、模型、注册、checkpoint、logging 等共享能力
  - 资源、设备、启动方式或其他运行前提

### 3. 把 branch diff 映射到 recipe 表面

- 读取目标 recipe 的主要入口和依赖面，而不是只盯着冲突文件本身。至少覆盖：
  - recipe 的 engine、model、data、config、入口脚本或注册点
  - recipe 对共享层能力的调用点
  - 如果存在，再读 recipe-local helper、schema、test 或验证脚本
- 找出 recipe 对上游共享变化和当前分支本地改动的直接依赖，以及这些依赖如何串成实际运行路径。
- 不要发现一个报错就立刻停下。先把 breakage list 和 merge hotspot list 建完整，再开始改。
- breakage mapping 不要只回答“哪个文件会冲突”，要回答：
  - 哪些能力会坏
  - 为什么会坏
  - 这些问题属于 recipe-local 适配、共享层缺陷，还是两边设计需要重新拼接
- 目标不是提前猜某一种已知 case，而是把 branch diff 转成一份清晰的 recipe adaptation 地图。

### 4. 先做 merge/adaptation 计划，再真正解决冲突

- 对每个 hotspot，先决定策略：
  - 直接接受上游版本
  - 保留当前分支的 recipe-local 设计
  - 手工组合两边改动
  - 识别为共享层通用问题并修共享代码
- 判断时优先看运行时语义和最终行为，而不是只看文本冲突块。
- 如果上游改的是共享契约，而当前分支改的是 recipe 接线，通常要保留两边并在 recipe 层做适配。
- 如果当前分支只是延续旧共享行为，而上游已经给出新的共享实现，不要把旧逻辑硬合回来；要明确迁移到新语义还是补必要兼容。

### 5. 在正确层次修 recipe 并解决 merge 冲突

- 优先在最小且正确的层次修问题：先修 recipe-local 适配，再看是否真的需要改共享层。
- 只有当问题明确属于共享实现、并且会影响多个 recipe 时，才修改 `mvp_engine/` 或其他共享代码。
- 不要过度抽象，也不要为了这次 merge 顺手做大规模重构。修复应保持直接、局部、可验证。
- 当分析已经充分后，再执行真正的 merge，优先使用：
  - `git merge --no-commit --no-ff <upstream_branch>`
- 先根据 hotspot map 解决冲突，再补必要的适配与修复。
- merge 完成前，不要因为“冲突消失了”就当成问题已解决；必须继续检查行为、契约和关键路径是否仍然成立。

### 6. 显式处理共享契约与本地假设之间的断层

- 逐项检查 recipe 对共享层的假设是否仍然成立，包括但不限于：
  - 输入输出约定
  - 配置与参数入口
  - 类、函数、hook、registry 或 helper 的调用方式
  - 初始化、加载、恢复、保存、评估等关键生命周期
  - 资源、设备、环境、权限或启动前提
- 不要只修表面的 import error 或冲突块；要确认 recipe 在真实运行路径上依赖的契约已经全部对齐。
- 如果上游改变了共享行为，而当前 recipe 仍依赖旧语义，需要明确决定：迁移到新语义、在 recipe 层补兼容，还是修共享层中的真实缺陷。
- 对模板型、可复用或对外暴露的 recipe，要额外检查常用入口、默认行为、扩展点和覆盖参数是否仍然可用，避免 merge 后出现“代码能跑但使用方式已悄悄失效”的情况。

### 7. 显式处理运行时与集成链路断层

- 检查 recipe 与周边系统的集成链路是否仍然连通，例如：
  - 数据读取与预处理
  - 模型构建、包装、权重加载、导出或恢复
  - 训练、评估、推理或工具脚本入口
  - checkpoint、logging、metrics、artifact 输出
  - 依赖的共享 helper、注册机制、启动逻辑或运行时能力
- 如果上游改动影响了调用顺序、默认行为、错误处理、资源假设或边界条件，需要把 recipe 的集成点一并修正，而不是只修单个函数签名。
- 对存在多条运行路径的 recipe，要确认主路径和常见辅助路径都没有被 merge 悄悄破坏。

### 8. 按目标路径验证 merge 结果

- 用能证明 breakage 已修好的最小充分验证层级，从轻到重逐步推进，例如：
  - 静态检查、导入、构建或最小初始化
  - 关键模块或关键路径 smoke test
  - 目标任务的启动验证
  - 更完整的训练、评估、推理或端到端验证
- 验证要覆盖这次 merge 实际影响到的路径，而不是只跑与变更无关的默认检查。
- 如果真实依赖不可用，不要直接跳过验证；优先构造最小可行 fixture、stub 或临时 override 去证明核心链路已经恢复。
- 临时验证手段不应永久削弱 recipe 默认行为；测试用调整和正式修复要区分开。

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
  - 两边重叠的热点文件、能力面和共享契约
- 确认 breakage list 和 merge hotspot list 已经覆盖这次 merge 实际影响到的关键路径。
- 确认 recipe 的实现、配置、入口和依赖链路都已经对齐 merge 后的共享契约，而不只是文本冲突被消掉。
- 对每一类 breakage 都执行对应的 targeted validation；验证方式应与实际受影响的能力相匹配。
- 验证可以从轻量检查逐步升级到更完整的运行验证，但至少要能证明核心链路已经恢复。
- 确认验证是在 merge 后的工作树上执行的，而不是 pre-merge 状态或局部手工拼接状态。
- 如果只做了部分验证，要明确说明：
  - 已验证了哪些路径
  - 还没有验证哪些路径
  - 剩余风险主要落在哪里
- 如果验证依赖额外环境、数据、权重、服务或计算资源，要明确说明这些前提，以及它们如何影响当前验证结论。

## Output

- 最终需要汇报：
  - 当前分支、上游分支、merge base
  - 检查过哪些 branch-only files / commits
  - 哪些文件、模块或契约是 merge hotspots
  - 在目标 recipe 中发现了哪些 breakages
  - 应用了哪些 recipe-local 修复或共享层修复
  - 关键 merge conflicts 是如何解决的
  - 跑了哪些验证、结果如何
  - 还剩哪些风险、限制或未验证路径
