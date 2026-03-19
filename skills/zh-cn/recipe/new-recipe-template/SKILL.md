---
name: new-recipe-template
description: 为本仓库在 recipes/ 下创建通用 recipe 脚手架。适用于用户要新建 recipe 起始文件，并且需要先询问 recipe 名称、任务简介和基础配置项，再生成目录与文件。
---

# new-recipe-template

## 目标

在 `recipes/<recipe_name>/` 下创建标准 recipe 目录：

- `README.md`
- `configs/`
- `dataset/`
- `model/`
- `engine/`
- `tests/`

实验特定逻辑必须留在 recipe 内，不要为单个 recipe 往 `mvp_engine/` 里加通用抽象。

## 1. 先把必要信息问清楚

先用一条简洁消息把关键信息问完，再开始生成：

- `snake_case` 的 recipe 名称
- 任务简介，用于 README 和 TODO 上下文
- 如果不想用 `train.yaml`，说明 config 文件名
- 是否生成 recipe-local tests

如果用户说“直接建一个”，默认值如下：

- task summary: `TODO: describe the task and training workflow.`
- config name: `train`
- include tests: `true`

命名规则：

- 目录名必须保持 `snake_case`
- 默认 engine class 名：`<RecipeNamePascalCase>Engine`

## 2. 用脚手架脚本生成

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

说明：

- 默认输出到 `recipes/`
- 验证脚手架时可先用 `--output-root /tmp/...`
- 只有明确要覆盖已有模板文件时才用 `--force`

## 3. 生成后必须人工检查

至少检查这些点：

- `project.name` 和 README 标题是否正确
- 确认生成的 config 仍然继承了仓库默认值，只做了必要的 recipe 定制
- engine class 与模块名是否和 recipe 名一致
- `dataset/` 和 `model/` 保持无实现代码状态，等真实逻辑确定后再写
- engine 里的大多数方法保持空实现并显式抛出 `NotImplementedError`
- README 要写成真实任务描述，而不是复制某个现成 recipe 的说法

如果后续需要具体实现模式，再去参考最接近的现有 recipe。不要把 `vit_classification` 或其他单一 recipe 直接当成默认模板。

## 4. 验证

至少执行：

```bash
python3 -m compileall recipes/<recipe_name>
```

然后优先执行：

```bash
uv run --with ruff ruff check recipes/<recipe_name>
```

如果生成了测试，再跑：

```bash
uv run --with pytest pytest -q recipes/<recipe_name>/tests
```

## 常见坑

- 不要把 recipe 专用 helper 挪进 `mvp_engine/`
- 不要生成 placeholder dataset/model 逻辑
- 不要为了让脚手架“更泛化”而过度抽象 engine
- 不要静默猜测某种模态或任务逻辑
- 对会 import `recipes.*` 的 recipe-local tests，不要漏掉 `tests/conftest.py`

## 参考

- 示例流程与生成树：`references/example.md`
