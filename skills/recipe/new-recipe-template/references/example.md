# Example

## Sample prompt flow

The skill should ask a compact question set before writing files. A good first message is:

```text
I can scaffold this recipe. Please confirm:
1. recipe name in snake_case
2. a short task summary for the README
3. config name if you do not want train.yaml
4. include recipe-local tests or not
```

## Sample command

```bash
python3 skills/recipe/new-recipe-template/scripts/create_recipe_template.py \
  --recipe-name tomato_baseline \
  --task-summary "Train and evaluate a new tomato recipe baseline." \
  --include-tests
```

## Expected tree

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

## Scope

The scaffold is intentionally general. It copies the repo's default config, leaves `dataset/` and `model/` without implementation code, and uses explicit `NotImplementedError` engine stubs. Existing recipes such as `vit_classification` can still be used later as references for concrete task wiring, but they should not define the default scaffold.
