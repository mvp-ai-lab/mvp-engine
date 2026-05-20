# MVP-Engine Skills Rules

- Follow repository-level `AGENTS.md` first; this file only defines additional rules for the `skills/` subtree.

## Skills Creation Rules

### Scope Boundaries

- Do not create a skill for one-off experiment logic that belongs in `recipes/`.
- Do not create a skill for stable generic runtime capabilities that belong in `mvp_engine/`.
- Create or extend a skill only when the task is a recurring agent workflow that does not fit a clean reusable API.
- Do not use a skill to restate repository-wide rules that belong in the root `AGENTS.md`.

### Decision Order

1. Check whether an existing skill already matches the task.
2. If not, check whether an existing skill can be extended without duplication.
3. Only create a new skill when the task is recurring, reusable, and not better represented in `mvp_engine/` or `recipes/`.

### Required Skill Shape

- Every skill must contain a `SKILL.md`.
- `SKILL.md` is the primary source of truth for that skill.
- Every `SKILL.md` should clearly state when the skill should be used.
- Every new skill must declare:
  - Goal
  - Required Inputs
  - Workflow
  - Validation
    - Soft Validation
    - Hard Validation (if applicable)
  - Output
  - Read on Demand
- Include references or examples when they are necessary for correct execution.
- When a skill changes a user recipe, the tests for that skill should live with
that recipe if they are required by `SKILL.md`, not under `skills/` and not under an unrelated demo recipe. Use this layout:

  ```text
  recipes/<recipe>/
  └── tests/
      ├── conftest.py
      ├── test_structure.py
      ├── test_smoke.py
      └── skills/
          └── <skill-id>/
              ├── asserts.py
              └── test_<impact>.py  # optional, only when the skill declares impact validation
  ```
- `tests/test_structure.py` and `tests/test_smoke.py` are recipe-level
  cumulative tests. They should be created when the recipe is created and
  extended only when the baseline recipe test surface must change. If they are missing, you should use the test template files under `tests/templates` to create them.
- If the skill requires hard validation, the skill directory must contain `asserts.py` Keep skill-specific structure and smoke assertions there.
- A skill directory may also contain optional `test_<impact>.py` files when the
  skill declares impact validation that structure and smoke cannot cover. Name
  the file after the measured impact, such as `test_memory_impact.py`,
  `test_compile_performance.py`, `test_behavior_parity.py`, or
  `test_checkpoint_compatibility.py`.

### Authoring Constraints

- Do not hardcode paths outside this repository inside a skill.
- Keep examples, templates, and references inside the skill's own folder.
- Keep skill instructions implementation-oriented, reusable, and specific enough for an agent to execute.

## Skills Using Rules

- When a user asks to use, execute, apply, validate, test, or run a skill for a recipe,
treat that request as an explicit request to use the fresh subagents required by
the repository skill-validation workflow.
- If the skill required by the user is conflicting with another skill that is already applied, stop and explain the conflict instead of applying the new skill.
- The coding agent must follow the skill instructions to apply the skill, and then run the required validation before declaring the skill application complete.

### Validation Rules

- Run these validation only in fresh subagents with `fork_context=false`.
Do not run them from the main agent's local terminal, background terminal sessions,
or any other non-subagent shell fallback. The validation must be in an isolated context.

- Validation is part of applying the skill and is a mandatory completion criterion,
not optional guidance. There are two types of validation:
  - Soft validation: the coding agent look through the skill instructions and the modified files, and validate that the implementation follows the instructions and does not introduce unintended changes. This MUST be done in a new subagent, just like a code reviewer.
  - Hard validation: the coding agent first generate the required pytest test files. Then run the required pytest tests in a fresh subagent. The coding agent MUST run all required tests, and stop on first failure.

#### Soft Validation

- You should only look through the code files to validate the implementation, not run any tests as the soft validation should be light-weight.

#### Hard Validation
- You should generate the required test files first if the skill requires hard validation. If the new skill conflicts with an existing assertion, the agent may update the assertion, but it must keep the smallest correct change and explain the conflict and resolution to the user.
- A skill application is incomplete until every required layer passes through
  the recipe-local pytest entrypoints, or the exact environment limitation and
  command to run in a real environment are reported.
- Tests must exercise the user's real recipe/model entrypoints with a minimal
  recipe-owned config or batch, not a separate toy model unrelated to that
  recipe.
- Use recipe-local pytest options such as `--world-size`,
  `--config-name`, and repeated `--config-override` flags if nessesary.
- If two skills require incompatible smoke configs, stop and explain the
  conflict instead of adding parallel smoke paths that hide the incompatibility.
- Impact validation is optional and only belongs in a skill when structure and
  smoke cannot verify the skill's expected effect. Impact tests may run multiple
  controlled jobs and compare measured metrics or invariants.

#### Typical validation workflow (run them in order, stopping on first failure)
1. do the soft validation first.
2. Structure test: `pytest recipes/<recipe>/tests/test_structure.py -q`: this makes sure the applied skill does not break the expected coding structure of the recipe, such as file locations, config structure, and engine structure.
3. Smoke test: it usually requires GPU/NPU resources, so first check your local environment or follow the instructions in `AGENTS.md` or `CUSTOM.md` to access the resources. Then run `pytest recipes/<recipe>/tests/test_smoke.py -q`: this makes sure the applied skill does not break the real training.
4. Optional impact validation: run each skill-declared `test_<impact>.py`, for
   example `pytest recipes/<recipe>/tests/skills/<skill-id>/test_memory_impact.py -q`.
