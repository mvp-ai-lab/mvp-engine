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

## Validation

在 `recipes/<recipe>/skill_tests/new-recipe-template/` 下补 recipe-local 测试：

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
- 优先先把 `tests/test_structure_template.py`、
  `tests/test_runtime_template.py`、`tests/test_smoke_template.py` 复制到
  recipe-local skill 目录，再只改 import 区块和最少量的 scaffold-specific 断言。
- 如果这个 skill 的 smoke 路径需要分布式执行，复制出来的 `test_smoke.py`
  应使用 `tests/test_smoke_template.py` 里的 `multi_rank_distributed_env(...)`，
  并根据 skill 要求或用户偏好，把运行模式配置成 DDP、FSDP2 shard、Tensor
  Parallel 或其他需要的分布式模式。
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
验证必须且只能交给全新的 subagent，并使用 `fork_context=false`。禁止主 agent
在本地终端、后台终端会话或其他任何非 subagent shell fallback 中直接运行这些
`python -m tests.test_skills` 命令。先启动一个 subagent 运行
`python -m tests.test_skills --recipe <recipe> --skill new-recipe-template --layer structure`，
只有它通过后，主 agent 才再启动新的 subagent 运行 `--layer runtime`；只有
runtime 通过后，主 agent 才再启动新的 subagent 运行 `--layer smoke`。最后由
主 agent 统一汇总三个层级的结果。如果 `test_smoke.py` 因 GPU、分布式启动条件
或执行权限受限而无法运行，主 agent 直接把准确的 `python -m tests.test_skills`
命令以及所需 launcher 命令返回给用户。

## Output

- 说明创建了哪个 recipe 路径。
- 说明使用了哪些默认值或用户指定选项。
- 总结哪些 placeholder 仍需要后续真实实现。
- 说明跑了哪些验证命令，哪些还没跑。

## Read On Demand

- 需要看期望的目录形态和示例工作流时，读取 `references/example.md`。
- 需要理解或调整脚本参数和输出行为时，读取 `scripts/create_recipe_template.py`。
