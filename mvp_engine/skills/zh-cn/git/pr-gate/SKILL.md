---
name: pr-gate
description: 用于在提交或更新 PR 前做质量门禁。适用于 commit 级别变更核查、按变更范围补充 docstring、以及统一执行 lint/test 的场景。
---

# Pre-PR Quality Gate

## 何时使用

- 用户要求 push 前或开 PR 前做代码质量整理。
- 用户要求按当前 commit 内容补 docstring/注释。

## 输入约定

- 目标分支（默认 `main`）。
- 检查范围：
  - 整个分支（`HEAD` vs `origin/<base>`）
  - 最近 N 个 commit
  - 指定 commit 列表
- 质量门禁命令（默认 `pre-commit run --all-files` + `pytest -q`，可按仓库调整）。

## 标准流程

1. 同步基线
- `git checkout <base>`
- `git pull --ff-only`
- 回到工作分支并确认状态干净。

2. 建立变更上下文
- 查看提交图：`git log --oneline --decorate --graph <base>..HEAD`
- 查看文件级改动：`git diff --name-status origin/<base>...HEAD`
- 对关键文件做逐段 diff 审阅。

3. docstring 补齐（按 commit 影响范围）
- 仅处理本分支触达到的函数、类、模块。
- 规则：
  - 新增公有函数/类必须有 docstring。
  - 修改签名、返回值、副作用、行为时必须更新 docstring。
  - 私有小函数若语义明显可不写。
- 禁止无信息描述，只写行为、输入输出、关键约束。

4. PR 前质量门禁
- 先跑格式化/静态检查，再跑测试。
- 失败时先修复与本次改动直接相关的问题。
- 记录未覆盖风险。

5. 产出
- 严重度排序的问题清单。
- 已执行命令与结果摘要。
- 建议提交信息。

## 输出模板

- Findings
  - `严重度 | 文件:行号 | 问题 | 建议`
- Validation
  - `命令 | 结果`
- Residual Risks
  - `未验证项`

## 需要参考时再读取

- [references/review-checklist.md](references/review-checklist.md): 高密度检查清单。
- [references/docstring-rules.md](references/docstring-rules.md): docstring 细则与示例。
