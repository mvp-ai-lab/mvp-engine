---
name: pr-skill-review
description: 对修改 skills/ 下文件的 PR 做评审。使用 skill 评审清单，指出具体问题与优点，并把反馈落到明确文件和章节上。
---

# 评审 Skill PR

## Goal

- 只评审 PR 中与 skill 相关的部分。
- 一致地应用仓库的 skill review checklist。
- 产出作者可以立刻执行的具体反馈。

## Required Inputs

- PR 上下文，例如 base/head 分支或 PR diff。
- 可选的聚焦路径或 skill 名称，如果用户只想看一部分。

## Workflow

### 1. 圈定 skill 改动范围

- 列出所有位于 `skills/` 下的改动文件。
- 如果 PR 还改了 `skills/` 之外的代码，除非用户要求扩展范围，否则只评审 skill 文件。

### 2. 应用 skill 评审清单

- 读取 [references/skill-review-checklist.md](references/skill-review-checklist.md)。
- 对每个被改动或新增的 skill，按以下维度检查：
  - 准确性
  - 完整性
  - 清晰度与一致性
  - 是否符合 skill 理念
  - 测试或验证指引
- 在评价工作流、示例或验证指引时，使用明确的章节引用。

### 3. 整理给作者的评审反馈

- 列出具体问题和建议，并注明文件与章节位置。
- 对已经写得好的内容也要简要指出，避免作者误改无问题部分。
- 每条反馈只说一个问题或建议，并给出推荐改法或必须澄清的问题。

## Validation

- 评审范围内的每个 skill 文件都实际检查过。
- 每条发现都引用了具体文件位置或章节。
- 评审结果能区分必须修改的问题和可以保留的内容。
- 反馈基于 checklist 维度，而不是泛泛的风格意见。

## Output

- Scope:
  - `已审文件: <skill 文件列表>`
- Issues / Suggestions:
  - `文件:行号 或 章节 | 问题或建议`
- Good As-Is:
  - 简要说明哪些内容应保持现状
- Summary:
  - 1-2 句整体评价和后续重点

## Read On Demand

- 在开始评审 skill 内容前，读取 [references/skill-review-checklist.md](references/skill-review-checklist.md)。
