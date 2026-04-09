---
name: pr-feedback
description: 用于 PR 已创建且收到评审意见后的处理。适用于分拣 reviewer 评论、做定向修复、补回归验证，并整理可直接回复 reviewer 的结论。
---

# 处理 PR 意见

## Goal

- 定向解决一个已打开 PR 上的 reviewer 反馈。
- 保持代码改动严格围绕评论意图，并让验证证据可直接复述给 reviewer。
- 产出可以直接发送的 reviewer 回复草稿。

## Required Inputs

- PR 上下文，例如 base/head 分支或等价的 diff 范围。
- reviewer 评论，包括行内评论、总结评论或关联 issue。
- 重新 push 前应执行的验证命令。

## Workflow

### 1. 收集评审上下文

- 收集所有未解决评论，并把每条评论映射到具体文件、行号或提交。
- 标记评论属于正确性阻塞、设计/可读性改进，还是仅需解释说明。

### 2. 分拣与计划

- 按处理动作分组：
  - 必须修复的正确性问题
  - 设计或可读性改进
  - 只需要解释的评论
- 对有冲突或歧义的评论，先标注为仍需用户或 reviewer 澄清。

### 3. 定向修复

- 每次改动都要与具体评论一一对应。
- 若行为契约变化，同步更新 docstring、类型标注或必要注释。
- 避免夹带与评论无关的清理性重构。

### 4. 回归验证

- 运行约定的 lint 和测试命令。
- 如果全量验证成本过高，就先跑能覆盖改动路径的目标检查，并显式说明剩余缺口。

### 5. 草拟 reviewer 回复

- 对每条评论说明：
  - 改了什么
  - 改在何处，使用 `文件:行号`
  - 用什么验证支持这次修改
- 如果不改代码，就给出简洁且技术上站得住的理由，不要含糊带过。

## Validation

- 每条未解决评论都被标记为 fixed、clarified 或 pending。
- 每个代码改动都能回溯到对应的 reviewer 评论。
- 验证结果已记录；如果仍有验证缺口，也已明确写出。
- 回复里引用的位置足够精确，reviewer 可以直接定位改动。

## Output

- Comment Resolution:
  - `comment id | action (fixed/clarified/pending) | 文件:行号`
- Validation:
  - `命令 | 结果`
- Pending Items:
  - `需要用户决策 / 需要 reviewer 确认`

## Read On Demand

- 当 PR 涉及 skill 内容时，读取 [references/feedback-checklist.md](references/feedback-checklist.md)，确保回复覆盖 skill reviewer 关心的维度。
- 当行为发生变化且触达代码需要同步更新文档或类型时，读取 [../pr-gate/references/docstring-and-typing.md](../pr-gate/references/docstring-and-typing.md)。
