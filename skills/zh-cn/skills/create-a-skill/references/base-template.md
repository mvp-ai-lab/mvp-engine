# 基础模板

当你需要从零创建一个新 skill 时，使用这个模板。替换占位符后，再删掉不需要的章节。

```md
---
name: <skill-name>
description: <这个 skill 做什么。什么时候应当使用它。>
---

# <Skill Title>

> 中文版：`skills/zh-cn/<category>/<skill-name>/SKILL.md`
> English version: `skills/en/<category>/<skill-name>/SKILL.md`

## Goal

- 说明目标。
- 说明边界。
- 说明预期交付物。

## Required Inputs

- 列出必需上下文。
- 列出必需文件、配置或约束。

## Workflow

### 1. Gather Context

- 说明先检查什么。
- 说明哪些决策依赖这些上下文。

### 2. Make Changes

- 说明主要实现路径。
- 说明常见变体如何分流。

## Validation

- 列出用于确认结果的检查项或命令。

## Output

- 说明最终回复必须包含什么。

## Read On Demand

- 指向可选的 `references/` 或 `scripts/` 条目，并说明何时使用。
```
