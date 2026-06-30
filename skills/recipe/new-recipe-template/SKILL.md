---
name: new-recipe-template
description: Create a new recipe scaffold under recipes/ with standard config,
  engine, dataset, model, README, and recipe-local test structure while keeping
  task-specific implementation intentionally minimal.
---

# New Recipe Template

## Goal

Create a clean recipe scaffold under `recipes/<recipe_name>/`:

- keep experiment-specific logic inside the recipe;
- create standard config, engine, dataset, model, and README files;
- include recipe-local `tests/test_structure.py` and `tests/test_smoke.py` when
  requested, and add `tests/test_contract.py` or impact/parity tests only when a
  skill or task needs those layers;
- leave task-specific data/model logic explicit and minimal until requirements
  are known;
- avoid adding repo-wide abstractions.

## Required Inputs

Collect these before writing files:

- recipe name in `snake_case`;
- short task summary;
- config filename, default `train`;
- whether to include recipe-local tests, default `true`;
- closest existing recipe to use as a reference, if any.

Ask once in a compact message when any required input is missing.

## Workflow

### 1. Validate Naming And Scope

Confirm:

- recipe folder is `snake_case`;
- default engine class is `<RecipeNamePascalCase>Engine`;
- config class is `<RecipeNamePascalCase>Config`;
- no existing recipe path will be overwritten.

### 2. Create Standard Layout

Create:

```text
recipes/<recipe_name>/
├── README.md
├── __init__.py
├── configs/
│   ├── __init__.py
│   ├── schema.py
│   └── <config_name>.yaml
├── dataset/
│   └── __init__.py
├── engine/
│   ├── __init__.py
│   └── <recipe_name>_engine.py
├── model/
│   └── __init__.py
└── tests/
    ├── test_structure.py
    └── test_smoke.py
```

Use `tests/templates/test_structure.py.template` and
`tests/templates/test_smoke.py.template` when creating baseline tests. Use
`tests/templates/test_contract.py.template` for fast semantic skill contracts and
`tests/templates/test_parity.py.template` for real impact/parity artifacts only
when the recipe needs those layers.

Read `references/scaffold_rules.md` before drafting files.

### 3. Keep Stubs Honest

Use explicit `NotImplementedError` for task-specific engine methods when the
real implementation is unknown. Do not guess dataset/model behavior.

The README and config should describe the provided task summary and clearly mark
remaining implementation work.

### 4. Validate The Scaffold

Run:

```bash
python3 -m compileall recipes/<recipe_name>
pytest recipes/<recipe_name>/tests/test_structure.py -q
```

Run smoke only after the recipe has real dataset/model/engine behavior.

## Validation

### Soft Validation

Review the scaffold without running tests:

- names and imports match the recipe name;
- config schema validates the YAML shape;
- engine is registered and names match;
- dataset and model remain minimal unless task-specific behavior is known;
- tests were created from current repo templates when requested;
- test layers stay within their boundaries: structure checks layout/config,
  contract checks cheap semantic invariants, smoke checks one-step runtime, and
  parity/impact checks real metrics;
- no `mvp_engine/` changes were introduced.

### Hard Validation

If recipe-local tests exist, optionally copy and adapt `references/asserts.py`
into:

```text
recipes/<recipe>/tests/skills/new-recipe-template/asserts.py
```

Run:

```bash
python3 -m compileall recipes/<recipe>
pytest recipes/<recipe>/tests/test_structure.py -q
```

Do not require `tests/test_smoke.py` to pass until real training behavior is
implemented.

## Output

- State recipe path created.
- State recipe name, config name, and whether tests were included.
- List files created.
- State placeholders or `NotImplementedError` methods that still need real
  implementation.
- State validation commands and results.

## Read On Demand

- `references/scaffold_rules.md`: standard file contents and scaffold choices.
- `references/example.md`: sample prompt flow and expected tree.
