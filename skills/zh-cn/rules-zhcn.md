# Skill 的统一标准

定义新增或修改 skill 时统一遵守的“标准模板”。

## 目录

- [Skill 的标准目录](#skill-layout)
- [`SKILL.md` 的强制元数据](#skill-frontmatter)
- [`SKILL.md` 的统一章节骨架](#skill-structure)
- [`references/` 的统一标准](#skill-references)
- [`scripts/` 的统一标准](#skill-scripts)
- [写作标准](#writing-guidelines)
- [基础模板](#base-template)

<a id="skill-layout"></a>

## Skill 的标准目录

skill 默认应同时维护英文和中文两个镜像目录：

```text
skills/
├── en/<category>/<skill-name>/
│   ├── SKILL.md
│   ├── references/   # 可选
│   └── scripts/      # 可选
└── zh-cn/<category>/<skill-name>/
    ├── SKILL.md
    ├── references/   # 可选
    └── scripts/      # 可选
```

约束如下：

- `<category>` 继续使用现有一级分类，例如 `training`、`parallel`、`model`、`git`
- `<skill-name>` 统一使用小写英文加连字符，例如 `gradient-checkpointing`
- `SKILL.md` 是必需文件
- `references/` 只在确实存在“按需阅读材料”时创建
- `scripts/` 只在确实存在“应由 skill 工作流调用的辅助脚本”时创建
- 不创建与执行无关的额外文档，例如 `README.md`、`CHANGELOG.md`
- 不提交 `__pycache__/`、临时文件、缓存文件、生成产物

<a id="skill-frontmatter"></a>

## `SKILL.md` 的强制元数据

每个 `SKILL.md` 顶部都必须包含 YAML front matter，且统一只保留两个字段：

```md
---
name: <skill-name>
description: <一句或两句，说明 skill 做什么，以及什么场景下应触发>
---
```

要求如下：

- `name` 必须与目录名一致
- `description` 必须同时覆盖“能力范围”和“触发场景”
- 不再在 front matter 中增加其他自定义字段
- 中英文版本的 `name` 必须一致
- 中英文版本的 `description` 可以分别用对应语言表达，但语义必须对齐

推荐写法：

- 先写 skill 做什么
- 再写什么时候使用
- 尽量让 agent 仅靠 `description` 就能判断是否应该触发这个 skill

<a id="skill-structure"></a>

## `SKILL.md` 的统一章节骨架

后续所有 skill 的主文档统一采用下面这套章节骨架。允许按 skill 类型裁剪，但章节命名尽量不要再发散。

```md
---
name: <skill-name>
description: <skill 能力 + 触发场景>
---

# <Skill Title>

> 中文版：`skills/zh-cn/<category>/<skill-name>/SKILL.md`
> English version: `skills/en/<category>/<skill-name>/SKILL.md`

## Goal

用 2 到 5 行说明这个 skill 的目标、边界和预期产物。

## Required Inputs

列出执行这个 skill 之前必须确认的上下文、输入文件、配置或约束。

## Workflow

### 1. <Step Name>

说明第一步做什么。

### 2. <Step Name>

说明第二步做什么。

## Validation

说明完成后如何验证，包括最小验证命令、检查项或通过标准。

## Output

说明最终交付给用户的输出应包含什么；如果需要固定回复格式，也放在这里。

## Read On Demand

列出什么时候需要去读 `references/` 下的哪些文件。
```

统一约束如下：

- 顶层目标章节统一使用 `Goal`
- 输入章节统一使用 `Required Inputs`
- 主流程章节统一使用 `Workflow`
- 验证章节统一使用 `Validation`
- 输出章节统一使用 `Output`
- 附加材料入口统一使用 `Read On Demand`

如果某类 skill 不需要某个章节，可删除，但不要随意改名为其他近义词，例如：

- 不再新增 `Steps`，统一并入 `Workflow`
- 不再新增 `Review output template`，统一并入 `Output`
- 不再新增 `Reference` 或 `Example` 作为主文档末尾章节，统一并入 `Read On Demand`

<a id="skill-references"></a>

## `references/` 的统一标准

`references/` 只放“按需阅读”的材料，不放主流程本身。

适合放进 `references/` 的内容：

- 参考实现
- 示例配置
- 样例测试
- 长篇说明
- 按框架、模型、后端拆分的变体资料

要求如下：

- `SKILL.md` 里必须明确说明“什么时候去读哪个 reference”
- reference 文件名应表达用途，例如 `vit_example.md`、`fsdp_notes.md`
- 如果 reference 很长，开头应提供目录或快速导航
- 不要把同一份说明同时写在 `SKILL.md` 和 `references/`

<a id="skill-scripts"></a>

## `scripts/` 的统一标准

`scripts/` 只放这个 skill 明确依赖的辅助脚本。

适合放进 `scripts/` 的情况：

- 同一段提取或转换逻辑会反复重写
- 这一步需要确定性，不能每次都临场生成
- 脚本可以显著降低主文档长度或减少出错概率

要求如下：

- `SKILL.md` 中必须明确脚本何时运行、输入是什么、输出是什么
- 脚本名称应直接表达用途
- 脚本应尽量保持单一职责
- 新增脚本后应至少做一次实际运行验证
- 如果脚本只服务某个语言版本，但逻辑是共享的，另一语言版本应引用同一脚本路径并说明原因

<a id="writing-guidelines"></a>

## 写作标准

后续所有 skill 的正文统一遵守这些写法：

- 以祈使句或操作指令句为主，不写成长篇背景介绍
- 优先写“怎么做”和“何时做”，少写 agent 已知的通用常识
- 主文档只保留核心流程，细节、示例、模板、样例配置放到 `references/`
- 当内容超过约 `300` 到 `500` 行时，优先拆到 `references/`
- 如果存在多种变体，主文档只保留分流规则，把变体细节拆到不同参考文件
- 标题层级最多建议到 `###`，避免过深嵌套
- 编号步骤统一使用顺序编号，不要同时混用多套编号体系

<a id="base-template"></a>

## 基础模板：

```md
---
name: <skill-name>
description: <这个 skill 做什么。什么时候应使用这个 skill。>
---

# <Skill Title>

> 中文版：`skills/zh-cn/<category>/<skill-name>/SKILL.md`
> English version: `skills/en/<category>/<skill-name>/SKILL.md`

## Goal

- 说明目标
- 说明边界
- 说明预期产物

## Required Inputs

- 列出必须确认的输入
- 列出必要前置条件

## Workflow

### 1. Gather Context

- 先收集什么信息
- 先检查什么文件

### 2. Make Changes

- 按什么原则修改
- 遇到什么情况如何分流

## Validate

### Validation Criteria / Test Cases 
- 跑什么命令
- 看什么结果

## Output

- 最终回复应包含什么
- 如果有固定模板，也放这里

## Read On Demand

- 什么时候读 `references/<file>.md`
- 什么时候跑 `scripts/<script>.py`
```
