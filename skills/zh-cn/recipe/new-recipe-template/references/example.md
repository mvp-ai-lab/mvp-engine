# 示例

## 推荐的首条提问

技能执行时，先用一条紧凑的问题把必要信息问清楚，例如：

```text
我可以先把 recipe 脚手架建出来。请确认：
1. recipe 名称（snake_case）
2. 一句话任务简介，用于 README
3. 如果不想用 train.yaml，请说明 config 名称
4. 是否生成 recipe-local tests
```

## 示例命令

```bash
python3 skills/en/recipe/new-recipe-template/scripts/create_recipe_template.py \
  --recipe-name tomato_baseline \
  --task-summary "Train and evaluate a new tomato recipe baseline." \
  --include-tests
```

## 预期目录

```text
recipes/tomato_baseline/
├── README.md
├── __init__.py
├── configs/
│   └── train.yaml
├── dataset/
│   ├── __init__.py
├── engine/
│   ├── __init__.py
│   └── tomato_baseline_engine.py
├── model/
│   ├── __init__.py
└── tests/
    ├── conftest.py
    └── test_tomato_baseline_scaffold.py
```

## 模板定位

这个脚手架本身是通用骨架。它会复制仓库默认 config，不生成 dataset/model 的实现代码，并让 engine 中的大多数方法显式抛出 `NotImplementedError`。后续如果需要具体实现模式，可以参考 `vit_classification` 等现有 recipe，但它们只是参考，不应定义默认脚手架。
