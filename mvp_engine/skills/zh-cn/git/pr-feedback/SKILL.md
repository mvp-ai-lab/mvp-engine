---
name: pr-feedback
description: 用于 PR 已创建且收到评审意见后的响应处理。覆盖评论分拣、定向修复、回归验证与 reviewer 回复草拟。
---

# 处理 PR 意见

## 何时使用

- 用户要求处理已打开 PR 的 reviewer 评论。
- 用户要求批量修复 review 反馈并准备回复。

## 输入约定

- PR 上下文（base/head 分支或等价 diff 范围）。
- reviewer 评论（行内评论、总结评论、关联 issue）。
- 回推前必须执行的验证命令。

## 标准流程

1. 收集评审上下文
- 拉齐所有未解决评论并按严重度/类型分组。
- 将每条评论映射到具体文件、行号、提交。

2. 分拣与计划
- 按处理动作分组：
  - 必须修复的正确性问题
  - 设计/可读性改进
  - 仅需解释说明
- 对有冲突或歧义的评论先标注待澄清。

3. 定向修复
- 每次改动严格围绕评论意图。
- 若行为契约变化，同步更新 docstring/注释。
- 避免夹带无关重构。

4. 回归验证
- 执行约定 lint/test 命令。
- 全量验证成本高时，先跑目标测试并显式报告缺口。

5. 草拟 reviewer 回复
- 每条评论给出：
  - 改了什么
  - 改在何处（`file:line`）
  - 用什么验证
- 若不改代码，给出简洁技术依据。

## 输出模板

- Comment Resolution
  - `comment id | action (fixed/clarified/pending) | 文件:行号`
- Validation
  - `命令 | 结果`
- Pending Items
  - `需要用户决策 / 需要 reviewer 确认`

## 需要参考时再读取

- [references/feedback-checklist.md](references/feedback-checklist.md): 当 PR 涉及 skill 时，回复前用此清单对照，确保覆盖 reviewer 关心的维度。
- [references/docstring-rules.md](references/docstring-rules.md): 行为变更时 docstring 更新规则。
