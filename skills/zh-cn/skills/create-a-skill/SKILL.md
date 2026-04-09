---
name: create-a-skill
description: 使用统一技能格式来新增或修改仓库内的 skill。适用于用户要求创建新 skill、重写现有 skill、或规范化某个 skill 目录及其中英文镜像时。
---

# Create or Update a Skill

> 中文版：`skills/zh-cn/skills/create-a-skill/SKILL.md`
> English version: `skills/en/skills/create-a-skill/SKILL.md`

## Goal

创建一个新 skill，或修订一个已有 skill，使其更易于被 agent 触发和执行。
让主文档 `SKILL.md` 只保留工作流、分流规则和验证标准。
当示例、模板或变体细节过长时，把它们移到 `references/`。

## Required Inputs

- 目标 skill 的路径，包括语言和分类。
- 如果是新 skill，还需要确定 `<category>` 和 kebab-case 的 `<skill-name>`。
- 这是新增 skill，还是修改已有 skill。
- 该 skill 要覆盖的用户任务，包括触发条件和预期输出。
- 是否已经存在 `skills/en/` 或 `skills/zh-cn/` 下的镜像目录。
- 该 skill 必须遵守的仓库约束或项目规则。

## Workflow

### 1. 先在内部判断这件事是否值得做成 skill

- 这一步只属于当前 `create-a-skill` 执行过程中的内部决策，不属于目标 skill 的内容。
- 不要把这段判断、它的理由，或“先向用户解释为什么该/不该做成 skill”的话术写进目标 skill 的 `SKILL.md`。
- 当任务有稳定流程，但实现必须按模型、recipe 或上下文做适配时，使用 skill。
- 如果逻辑本应沉淀为通用复用 API，就不要做成 skill。
- 如果只是一次性实验胶水代码，应留在 `recipes/`，不要强行抽成 skill。
- 如果结论是否定的，直接在当前回复中说明原因，并停止创建或修改目标 skill 文件。

### 2. 先定义这个 skill 的契约

- 先确定分类和 kebab-case 的 skill 名称，再开始写文件。
- 先写一个简短的 `description`，同时说明能力范围和触发场景。
- 先定义预期输出、验证标准，以及是否真的需要 `references/` 或 `scripts/`。
- 如果是更新已有 skill，除非用户明确要求重命名，否则保持原有 `name` 不变。

### 3. 按起始状态分流

- 如果是新 skill：
  - 在 `skills/en/` 和 `skills/zh-cn/` 下创建镜像目录
  - 基于基础模板起草 `SKILL.md`
  - 只有在工作流确实需要时，才添加 `references/` 或 `scripts/`
- 如果是已有 skill：
  - 阅读现有 `SKILL.md`、镜像文件，以及本地的 `references/` 或 `scripts/`
  - 识别结构问题：
    - 缺少 YAML front matter
    - `name` 与目录名不一致
    - `en/` 或 `zh-cn/` 下语言放错
    - 细节堆在主文档里，本应移到 `references/`
    - 缺少验证标准或输出要求
  - 保留有价值的信息，但删掉漂移、重复和猜测性表述

### 4. 规范目录结构

- 默认维护中英文镜像目录：
  - `skills/en/<category>/<skill-name>/`
  - `skills/zh-cn/<category>/<skill-name>/`
- `SKILL.md` 为必需文件。
- 只有在确实存在按需阅读材料时才创建 `references/`。
- 只有在工作流明确依赖确定性辅助脚本时才创建 `scripts/`。
- 不要添加 `README.md`、`CHANGELOG.md` 这类无关文件。

### 5. 按统一格式编写或重写 `SKILL.md`

- 文档顶部必须是 YAML front matter，且只保留这两个字段：
  - `name`
  - `description`
- `name` 必须与目录名一致，并在中英文镜像中保持完全相同。
- `description` 必须同时覆盖能力范围和触发场景。
- 除非某节确实不需要，否则统一使用以下章节骨架：
  - `Goal`
  - `Required Inputs`
  - `Workflow`
  - `Validation`
  - `Output`
  - `Read On Demand`
- 以直接操作指令为主，不写长篇背景介绍。
- 主文档只保留决策规则、执行步骤和通过标准。
- 目标 skill 只描述“这个 skill 被触发后该怎么执行任务”；不要加入“先判断这件事是否应该做成 skill”之类的元步骤。

### 6. 只在确实有价值时拆分附加材料

- 把基础模板、长示例、测试样板、按变体拆分的细节移到 `references/`。
- 在 `SKILL.md` 中明确说明每个 reference 何时需要打开。
- 只有当某一步反复手写或明显容易出错时，才新增脚本。
- 如果新增脚本，必须说明输入、输出以及运行时机。

### 7. 保持镜像对齐

- `skills/en/` 下放英文内容，`skills/zh-cn/` 下放中文内容。
- 中英文镜像的结构和语义必须对齐。
- 翻译不必逐字对应，但工作流、分流规则和验证标准必须一致。

## Validation

- 确认最终 `SKILL.md` 的 front matter 只包含 `name` 和 `description`。
- 确认新 skill 已同时创建中英文镜像目录，并包含必需的 `SKILL.md`。
- 确认主文档使用统一章节名；如果省略某节，必须是因为它确实不需要。
- 确认 `references/` 下的条目都在 `Read On Demand` 中被明确引用。
- 确认 `scripts/` 下的条目都在工作流中被显式调用，并写明输入和输出。
- 确认 `en` 与 `zh-cn` 镜像使用了正确语言，且语义保持对齐。
- 确认生成或更新后的目标 skill 不包含“先判断是否应该做成 skill”之类的元步骤，也不要求后续 agent 先向用户解释这类内部判断。
- 如果新增或修改了脚本，至少实际运行一次。

## Output

- 说明创建或更新了哪些 skill 路径。
- 说明这次产出的是全新 skill、对已有 skill 的更新，还是两者都有。
- 总结重要的结构性修改，不要逐句解释措辞变化。
- 说明做了哪些验证，哪些没有验证。
- 如果仍有刻意保留的限制或后续工作，明确列出。

## Read On Demand

- 当你要从零创建新 skill，或要对一个质量较差的旧 skill 做整体验证式重写时，读取 [references/base-template.md](/home/c84391361/projects/mvp-engine/skills/zh-cn/skills/create-a-skill/references/base-template.md)。
- 只有当任务要求同步更新目标 skill 下的 `references/` 或 `scripts/` 时，才继续读取这些文件。
