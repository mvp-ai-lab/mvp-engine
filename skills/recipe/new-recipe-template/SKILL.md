---
name: new-recipe-template
description: Create a new recipe scaffold under recipes/ in this repo. Use when the user wants starter files for a new recipe and you need to collect the missing recipe name, task summary, config name, or test choice before generating files.
---

# New Recipe Template

## Goal

- Create a new recipe folder under `recipes/<recipe_name>/` with the standard repo layout.
- Keep experiment-specific logic inside the recipe instead of adding repo-wide abstractions.
- Leave dataset and model implementation intentionally empty until the real task-specific logic is known.

## Required Inputs

- The recipe name in `snake_case`.
- A short task summary for the README and scaffold context.
- The config filename when `train.yaml` is not desired.
- Whether recipe-local tests should be generated.

Reasonable defaults when the user says to just scaffold it:
- task summary: `TODO: describe the task and training workflow.`
- config name: `train`
- include tests: `true`

## Workflow

### 1. Collect the missing inputs first

- Ask for the recipe name, task summary, config filename, and whether to include tests before generating files.
- Ask once in a compact message instead of drip-feeding questions.
- Keep naming rules explicit:
  - the folder name must stay `snake_case`
  - the default engine class is `<RecipeNamePascalCase>Engine`

### 2. Generate the scaffold with the shared script

Use the shared script:

```bash
python3 skills/recipe/new-recipe-template/scripts/create_recipe_template.py \
  --recipe-name <recipe_name> \
  --task-summary "<short summary>"
```

Common optional flags:

```bash
python3 skills/recipe/new-recipe-template/scripts/create_recipe_template.py \
  --recipe-name <recipe_name> \
  --task-summary "<short summary>" \
  --config-name train \
  --include-tests
```

- The script defaults to `recipes/` as the output root.
- Use `--output-root /tmp/...` when validating a scaffold without touching the repo tree.
- Use `--force` only when intentionally overwriting an existing scaffold file.

### 3. Review the generated recipe before stopping

- Inspect the generated files and tighten obvious placeholders.
- Confirm:
  - `project.name` and the README title match the recipe name
  - the config still follows repo defaults except for intended recipe-local overrides
  - engine class and module names match the recipe name
  - `dataset/` and `model/` remain implementation-free until the real logic is ready
  - engine methods stay explicit and empty rather than guessing task-specific behavior
  - the README describes the real task instead of a copied example
- If the user later needs a concrete implementation pattern, use the closest existing recipe as a reference after the scaffold exists.

### 4. Validate the scaffold

Run at least:

```bash
python3 -m compileall recipes/<recipe_name>
```

Prefer to run:

```bash
uv run --with ruff ruff check recipes/<recipe_name>
```

## Validation

Create the recipe-level tests and initial skill assertions:

- `skill_tests/test_structure.py`: verify recipe structure and core wiring.
- `skill_tests/test_smoke.py`: run one real recipe-owned training step and checkpoint/log path.
- `skill_tests/new-recipe-template/asserts.py`: keep the new-recipe-template
  assertions in the standard `assert_structure(...)` and `assert_smoke(...)`
  hooks.

## Output

- State which recipe path was created.
- State which defaults or user-provided options were used.
- Summarize any placeholder content that still needs real implementation.
- State which validation commands ran and which did not.

## Read On Demand

- Read `references/example.md` when you need the expected scaffold shape and a sample workflow.
- Read `scripts/create_recipe_template.py` when you need to understand or adjust the script's flags and output behavior.
