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

在 `recipes/<recipe>/skill_tests/new-recipe/` 下补 recipe-local 测试：

- `test_spec.yaml`：声明这个 skill 在该 recipe 上要求哪些测试层级。
- `test_structure.py`：至少验证 recipe import、registry 接线、config schema 校验、
  required slots，以及 logger/checkpoint hooks；对这个 scaffold skill 还必须验证
  生成出的目录结构存在、预期文件已创建、包名与模块名和 recipe 名一致，以及
  README / config 中的占位内容已按用户请求改写。
- `test_runtime.py`：至少成功构建 dataset、collator、model、optimizer、
  scheduler 和 engine，且不直接启动训练；对这个 scaffold skill 还必须验证
  recipe 模块可正常 import、config schema 可通过校验、engine class 以配置
  里的名字完成注册，并且这套 scaffold 接线可被解析。
- `test_smoke.py`：覆盖 1 个真实、recipe-owned 的 single step：forward、loss、
  backward、optimizer step、logger write，以及 checkpoint noop 或临时保存；
  使用 scaffold 自己的入口，并把配置或 batch 缩到仍能证明 scaffold 接通正确
  的最小规模。
- `test_smoke.py` 必须走该 skill 的完整真实能力路径：真实 scaffold recipe 入口、
  真实 engine 接线，以及真实 logger / checkpoint 行为；禁止用 monkeypatch、
  fake engine、fake training step 或类似测试桩把要验证的能力短路掉。
- 如果该 recipe 的 full-capability single-step 只能在 GPU 或分布式环境下成立，
  就把 smoke test 写成真实 launcher 测试，并在 `test_spec.yaml` 里把
  `gpu_preferred` 设为 `true`；不要为了在更弱环境里跑通而退化成 fake 逻辑。

这些 skill tests 与 scaffold 自带的普通 `tests/` 目录是分开的。它们应聚焦在
脚手架正确性上，而不是尚未实现的任务训练行为。

不要换成与该 recipe 无关的 toy recipe 或 toy model。应直接使用用户新建的
recipe package、config 和 engine 入口，只把验证路径缩到仍能覆盖 scaffold
落点的最小规模。

当你在用户 recipe 上执行这个 skill 时，应默认自动补齐这些测试，不要要求用户自己列出测试文件。
如果因为 GPU、分布式启动条件或执行权限受限而无法运行，直接把准确的 `python -m tests.test_skills`
命令以及所需 launcher 命令返回给用户。

## 常见坑

- 不要把 recipe 专用 helper 挪进 `mvp_engine/`
- 不要生成 placeholder dataset/model 逻辑
- 不要为了让脚手架“更泛化”而过度抽象 engine
- 不要静默猜测某种模态或任务逻辑
- 对会 import `recipes.*` 的 recipe-local tests，不要漏掉 `tests/conftest.py`

## 参考

- 示例流程与生成树：`references/example.md`
