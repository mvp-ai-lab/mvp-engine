---
name: new-recipe-template
description: 为本仓库在 recipes/ 下创建新的 recipe 脚手架。适用于用户要新建 recipe 起始文件，并且需要先收集 recipe 名称、任务简介、配置文件名或测试选项后再生成目录与文件。
---

# New Recipe Template

## Goal

- 在 `recipes/<recipe_name>/` 下创建标准 recipe 脚手架。
- 把实验特定逻辑留在 recipe 内，而不是顺手加到仓库级抽象里。
- 在真实任务逻辑明确前，让 dataset 和 model 目录保持空实现状态。

## Required Inputs

- `snake_case` 的 recipe 名称。
- 用于 README 和脚手架上下文的简短任务简介。
- 如果不想用 `train.yaml`，说明 config 文件名。
- 是否需要生成 recipe-local tests。

当用户说“直接建一个”时，可用默认值：
- task summary: `TODO: describe the task and training workflow.`
- config name: `train`
- include tests: `true`

## Workflow

### 1. 先把缺失输入问清楚

- 在开始生成前，先确认 recipe 名称、任务简介、config 文件名和是否包含测试。
- 尽量用一条简洁消息问完，不要拆成很多轮零散问题。
- 命名规则要明确：
  - 目录名必须保持 `snake_case`
  - 默认 engine class 名为 `<RecipeNamePascalCase>Engine`

### 2. 用共享脚本生成脚手架

使用共享脚本：

```bash
python3 skills/en/recipe/new-recipe-template/scripts/create_recipe_template.py \
  --recipe-name <recipe_name> \
  --task-summary "<short summary>"
```

常用可选参数：

```bash
python3 skills/en/recipe/new-recipe-template/scripts/create_recipe_template.py \
  --recipe-name <recipe_name> \
  --task-summary "<short summary>" \
  --config-name train \
  --include-tests
```

- 脚本默认输出到 `recipes/`。
- 如果只是想先验证生成效果，可用 `--output-root /tmp/...`。
- 只有明确需要覆盖已有脚手架文件时才使用 `--force`。

### 3. 停止前人工检查生成结果

- 检查生成后的文件，并收紧明显的 placeholder。
- 至少确认：
  - `project.name` 和 README 标题与 recipe 名称一致
  - config 仍然遵循仓库默认值，只保留必要的 recipe 特化
  - engine class 和模块名与 recipe 名一致
  - `dataset/` 和 `model/` 在真实逻辑确定前保持无实现状态
  - engine 方法保持显式空实现，而不是猜测任务逻辑
  - README 描述的是实际任务，而不是拷贝来的现成例子
- 如果后续用户需要具体实现模式，在脚手架创建完成后再参考最接近的现有 recipe。

### 4. 验证脚手架

至少执行：

```bash
python3 -m compileall recipes/<recipe_name>
```

更推荐执行：

```bash
uv run --with ruff ruff check recipes/<recipe_name>
```

如果生成了测试，再执行：

```bash
uv run --with pytest pytest -q recipes/<recipe_name>/tests
```

## Validation

- 生成后的目录树包含预期的 recipe-local 文件和目录。
- recipe 名称是 `snake_case`，engine class 使用匹配的 PascalCase 形式。
- 在已知信息足够的地方，placeholder 文本已被收紧。
- `dataset/` 和 `model/` 仍然保持有意的空实现状态。
- 生成出的 recipe 已经过编译检查，并在可行时跑过 lint 和 smoke test。

## Output

- 说明创建了哪个 recipe 路径。
- 说明使用了哪些默认值或用户指定选项。
- 总结哪些 placeholder 仍需要后续真实实现。
- 说明跑了哪些验证命令，哪些还没跑。

## Read On Demand

- 需要看期望的目录形态和示例工作流时，读取 `references/example.md`。
- 需要理解或调整脚本参数和输出行为时，读取 `scripts/create_recipe_template.py`。
