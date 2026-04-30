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
python3 skills/en/recipe/new-recipe-template/scripts/create_recipe_template.py \
  --recipe-name <recipe_name> \
  --task-summary "<short summary>"
```

Common optional flags:

```bash
python3 skills/en/recipe/new-recipe-template/scripts/create_recipe_template.py \
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

Add recipe-local tests under `recipes/<recipe>/skill_tests/new-recipe-template/`:

- `test_spec.yaml`: declare the required test layers for this applied skill.
- `test_structure.py`: at least verify recipe import, registry wiring, config
  schema validation, required slots, and logger/checkpoint hooks; for this
  scaffold skill it must also verify the generated layout exists, expected files
  are created, package/module names match the recipe name, and the README/config
  placeholders were rewritten for the requested recipe.
- `test_runtime.py`: at least build dataset, collator, model, optimizer,
  scheduler, and engine successfully without starting training; for this
  scaffold skill it must also verify the recipe modules import cleanly, the
  config schema validates, the engine class is registered under the configured
  name, and the scaffold wiring can be resolved.
- `test_smoke.py`: cover one real recipe-owned single step: forward, loss,
  backward, optimizer step, logger write, and checkpoint noop or temporary
  save, using the scaffold's own entrypoints with the smallest recipe-owned
  config or batch that still proves the scaffold is connected correctly.
- Prefer copying `tests/test_structure_template.py`,
  `tests/test_runtime_template.py`, and `tests/test_smoke_template.py` into the
  recipe-local skill directory first, then only edit the import block and the
  minimum scaffold-specific assertions you need.
- If this skill's smoke path needs distributed execution, the copied
  `test_smoke.py` should use `multi_rank_distributed_env(...)` from
  `tests/test_smoke_template.py` and configure the run as DDP, FSDP2 sharding,
  tensor parallel, or another required mode based on the skill requirement or
  user preference.
- `test_smoke.py` must use the full real capability path for this skill: real
  scaffold recipe entrypoints, real engine wiring, and real logger / checkpoint
  behavior. Do not short-circuit it with monkeypatch-based fake engines, fake
  training steps, or similar test-only stand-ins.
- If the recipe's full-capability single step only makes sense on GPU or
  distributed hardware, write the smoke test as a real launcher-driven smoke
  test and set `gpu_preferred: true` in `test_spec.yaml`; do not degrade it
  into fake logic just to make it run in a weaker environment.

These skill tests are separate from the scaffold's normal `tests/` directory. Keep
them focused on scaffold correctness, not task-specific training behavior that does
not exist yet.

Do not swap in an unrelated toy recipe or model for this skill. Use the user's new
recipe package, config, and engine entrypoints directly, with the smallest
recipe-owned validation path that still exercises the scaffold landing points.

When executing this skill for a user recipe, add these tests automatically. Do not
require the user to spell out the test file list. Run validation only in fresh
subagents with `fork_context=false`. Do not run these `python -m tests.test_skills`
commands from the main agent's local terminal, background terminal sessions, or
any other non-subagent shell fallback. First run
`python -m tests.test_skills --recipe <recipe> --skill new-recipe-template --layer structure`,
then a new subagent for `--layer runtime` only after structure passes, and then a
new subagent for `--layer smoke` only after runtime passes. The main agent should
summarize all three layer results. If `test_smoke.py` is blocked by GPU
availability, distributed-launch constraints, or permissions, the main agent
should return the exact `python -m tests.test_skills` command and any required
launcher command for the user.

## Output

- State which recipe path was created.
- State which defaults or user-provided options were used.
- Summarize any placeholder content that still needs real implementation.
- State which validation commands ran and which did not.

## Read On Demand

- Read `references/example.md` when you need the expected scaffold shape and a sample workflow.
- Read `scripts/create_recipe_template.py` when you need to understand or adjust the script's flags and output behavior.
