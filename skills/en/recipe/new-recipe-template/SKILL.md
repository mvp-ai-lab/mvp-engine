---
name: new-recipe-template
description: Create a new general recipe scaffold under recipes/ in this repo. Use when the user wants starter files for a new recipe, and first ask for the recipe name, task summary, and basic config choices before generating the files.
---

# new-recipe-template

## Goal

Create a new recipe folder under `recipes/<recipe_name>/` with the standard repo layout:

- `README.md`
- `configs/`
- `dataset/`
- `model/`
- `engine/`
- `tests/`

Keep experiment-specific logic inside the recipe. Do not add repo-wide abstractions for one recipe.

## 1. Ask for the missing inputs first

Ask once, in a compact plain-text message. Do not start generating files until the following are clear:

- recipe name in `snake_case`
- short task summary for the README and TODO context
- config name if `train.yaml` is not desired
- whether to include recipe-local tests

Reasonable defaults if the user says "just scaffold it":

- task summary: `TODO: describe the task and training workflow.`
- config name: `train`
- include tests: `true`

Naming rules:

- folder name must stay `snake_case`
- default engine class is `<RecipeNamePascalCase>Engine`

## 2. Generate with the scaffold script

Use the shared script:

```bash
python3 skills/en/recipe/new-recipe-template/scripts/create_recipe_template.py \
  --recipe-name <recipe_name> \
  --task-summary "<short summary>"
```

Useful optional flags:

```bash
python3 skills/en/recipe/new-recipe-template/scripts/create_recipe_template.py \
  --recipe-name <recipe_name> \
  --task-summary "<short summary>" \
  --config-name train \
  --include-tests
```

Notes:

- The script defaults to `recipes/` as the output root.
- Use `--output-root /tmp/...` when validating the scaffold without touching the repo tree.
- Use `--force` only when you intentionally want to overwrite an existing scaffold file.

## 3. Review the generated recipe before stopping

Always inspect the generated files and tighten obvious placeholders:

- set `project.name` and README title correctly
- confirm the generated config still matches repo defaults except for the intended recipe-specific overrides
- confirm the engine class and module names match the recipe name
- keep `dataset/` and `model/` code-free until the real implementation is ready
- keep engine methods explicit and empty instead of guessing task-specific logic
- make the README reflect the actual task, not a copied example

If the user later needs a concrete pattern, use the closest existing recipe as a reference after the scaffold exists. Do not hard-code `vit_classification` or any other single recipe as the scaffold itself.

## 4. Validate

At minimum run:

```bash
python3 -m compileall recipes/<recipe_name>
```

Then prefer:

```bash
uv run --with ruff ruff check recipes/<recipe_name>
```

If tests were generated, run the recipe-local smoke test:

```bash
uv run --with pytest pytest -q recipes/<recipe_name>/tests
```

## Pitfalls

- Do not move recipe-only helpers into `mvp_engine/`.
- Do not generate placeholder dataset/model logic.
- Do not over-abstract the engine just to make the scaffold look generic.
- Do not silently guess modality-specific code.
- Do not omit `tests/conftest.py` for recipe-local tests that import `recipes.*`.

## Reference

- Example workflow and generated tree: `references/example.md`
