---
name: pr-skill-review
description: 对「修改 skill 文件」的 PR 做评审。应用 skill 评审清单，输出具体问题、建议与可保留项。
---

# 评审 Skill PR

## 何时使用

- 用户要求你对「修改了 `skills/` 下文件」的 PR 做 review（如 SKILL.md、references/*.md）。
- 用户要求针对 skill 内容做评审（模式、完整性、清晰度、是否符合 skill 理念、测试指引）。

## 输入约定

- PR 上下文（base/head 分支或 PR diff）。
- 可选：指定要重点看的 skill 或路径。

## 标准流程

1. **识别 skill 相关改动**
   - 列出 PR 中所有位于 `skills/` 下的文件（任意语言）。
   - 若 PR 同时改代码与 skill，可将评审范围限定为 skill 相关文件（或与用户约定只审 skill）。

2. **应用 skill 评审清单**
   - 阅读并应用 [references/skill-review-checklist.md](references/skill-review-checklist.md)（与 `pr-feedback` 的 feedback-checklist 维度一致）。
   - 对每个被改动的 skill（或新增 skill）逐项检查：准确性、完整性、清晰与一致、与 skill 理念一致、测试指引。
   - 评论时可使用清单中的「Skill 位置」引用主流程、示例或测试模板路径。

3. **给作者的输出**
   - 列出**具体问题与建议**，并注明文件与章节（如 `path/to/SKILL.md § 何时使用`）。
   - 若有**可保留**之处，简要说明以便作者知道哪些无需改。
   - 反馈要可执行：一点一条、带位置与建议改法或待澄清问题。

## 输出模板

- **范围**
  - `已审文件: <skill 文件列表>`
- **问题 / 建议**
  - `文件:行号 或 § 章节 | 问题或建议`
- **可保留**
  - 简要说明哪些保持现状即可。
- **小结**
  - 1～2 句：整体评价与主要后续动作。

## 需要参考时再读取

- [references/skill-review-checklist.md](references/skill-review-checklist.md): **评审** skill 内容时使用的完整清单（准确性、完整性、清晰、理念、测试）。
