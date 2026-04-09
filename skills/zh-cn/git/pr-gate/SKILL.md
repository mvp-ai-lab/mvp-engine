---
name: pr-gate
description: 用于 push 前或开关 PR 前的质量门禁。适用于检查分支改动、补齐触达 API 的 docstring 与 typing、执行约定的 lint 和测试，并总结剩余风险。
---

# Pre-PR Quality Gate

## Goal

- 在 push 或更新 PR 前审视当前分支的代码质量。
- 对触达的公有 API 补齐或修正文档字符串与类型标注。
- 跑完约定的 lint 和测试门禁，并明确剩余风险。

## Required Inputs

- 目标基线分支，通常是 `main`。
- 要检查的范围：
  - 整个分支（`HEAD` vs `origin/<base>`）
  - 最近 N 个 commit
  - 指定 commit 列表
- 质量门禁命令。默认使用 `pre-commit run --all-files` 和 `pytest -q`，除非仓库需要更窄的命令集合。

## Workflow

### 1. 同步基线

- 更新所选 base 分支的本地基线。
- 回到工作分支后确认工作区状态，再开始看 diff。

### 2. 建立变更上下文

- 查看目标范围内的提交图。
- 建立文件级改动映射。
- 对关键文件阅读实际 diff，而不是只看文件名。

### 3. 修正文档字符串与类型标注

- 只处理当前分支已经触达的函数、类和模块。
- 规则如下：
  - 新增公有函数和类必须有 docstring
  - 新增或修改过的公有函数，在语言支持时应补齐参数和返回值类型标注
  - 当签名、返回值、副作用或行为变化时，docstring 与类型标注必须一起更新
  - 对已触达代码中的过时或错误类型做收敛；能写具体类型就不要保留宽泛 `Any`
  - 语义明显的私有小函数可以不写 docstring
- 保持文档与真实行为一致，避免无信息量描述。

### 4. 运行质量门禁

- 先跑格式化和静态检查，再跑测试。
- 优先修复与当前改动直接相关的失败。
- 如果无法承担全量门禁成本，就运行最有说服力的目标检查，并明确剩余覆盖缺口。

### 5. 准备可用于 PR 的摘要

- 按严重度排序整理发现项。
- 记录每个已执行命令及其结果。
- 如果用户需要，补一个简短的建议提交信息。

## Validation

- 审查范围与用户要求的 base 分支和 commit 范围一致。
- 触达的公有 API 具有与真实行为一致的 docstring 和类型标注。
- 质量门禁命令已经执行；若未执行，也给出了明确原因。
- 只有真正未覆盖的区域才被列为残余风险。

## Output

- Findings:
  - `严重度 | 文件:行号 | 问题 | 建议`
- Validation:
  - `命令 | 结果`
- Residual Risks:
  - `未验证项`
- Suggested Commit Message:
  - 需要时给出简短祈使句摘要

## Read On Demand

- 当触达代码需要补 docstring 或类型标注时，读取 [references/docstring-and-typing.md](references/docstring-and-typing.md)。
