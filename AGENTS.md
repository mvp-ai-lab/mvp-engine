# MVP-Engine Development Rules

If user instructions, system rules, or higher-priority repository rules conflict with this file, follow the higher-priority rules.

## Reading Order

Priority reading order:

1. Read `AGENTS.md` files in the current directory and ancestor directories first
2. If `CUSTOM.md` exists in the repository, read it as well

## Project Structure & Module Organization
- `mvp_engine/`: core package.
- `mvp_engine/engine/`: training orchestration and engine base classes.
- `mvp_engine/dataset/`: dataset builders and data pipeline utilities.
- `mvp_engine/distributed/` and `mvp_engine/utils/`: distributed/runtime helpers and other utilities.
- `skills/`: agent skills — structured guides for tasks that have clear patterns but cannot be generalized into a single API (for example gradient checkpointing, FSDP wrap policies). Organized by category (`training/`, `parallel/`, `model/`, `data/`, `debug/`, `recipe/`, `experiment/`, `config/`, `git`, `skills/`). See `skills/README.md` for overview and design rationale.
- `recipes/`: experiment-specific engines, models, datasets, and Hydra YAML configs (for example `recipes/vit_classification/configs/`).
- `tests/`: pytest suite (`test_*.py`) and shared fixtures (`conftest.py`).
- `tools/dataviewer/`: local data viewer app.
- `assets/`, `data/`, `outputs/`, `pretrained/`: static assets, local data links, run artifacts, and model weights.

## Repository Purpose

This repository contains the core engine, shared utilities, and experiment-specific configurations for training multimodal models. The main components are:

- `mvp_engine/` for stable, generic, reusable training infrastructure, including:
  - Launch entrypoints, default configs, basic training engine, distributed initialization tools, logging tools, and checkpoint infrastructure.
  - Treat it as stable core by default; do not change it casually
- `recipes/` for experiment-specific logic, including:
  - Configs, datasets, and training workflows for specific experiments
- `skills/` for structured agent-facing instructions, including:
  - Instructions for how to add new capabilities to the system, and how to maintain the repository

## Code Styles

- Use Python 3.12
- 4-space indentation
- Maximum line length 120
- Use `snake_case` for functions, modules, and files
- Use `PascalCase` for classes
- Avoid deprecated APIs
- Code should be clean, minimal, and easy to maintain
- All public class/functions/methods should have clear, descriptive names and docstrings, but internal helper functions can be more concise and may omit docstrings if their purpose is clear from context
- Prefer simple, direct solutions
- Avoid unnecessary helper functions
- Do not over-engineer
- Avoid unnecessary abstractions
- Be extremely concise
- No explanations unless explicitly requested
- Prefer diff format or partial snippets over full files
- Do not introduce new dependencies unless required
- Do not refactor unless necessary
- Do not add features not requested
- Do not touch unrelated files
- Fix only the exact problems

## High-priority working rules:
- Must put experiment-specific logic in `recipes/<experiment>/`
- Try not to modify `mvp_engine/`. If you need to, ask the user for confirmation first.
- Check whether the repository already provides the needed capability/infrastructure before writing new code. The provided capability may be in `skills/` as a skill, or in `mvp_engine/` as a core component.
- If the correct entrypoint, config, workflow, or module is unclear, inspect the repository first and then confirm with the user.

## A usual workflow

1. After the user states a need, the very first step is to check whether `skills/` and `mvp_engine/` already contains a matching skill/component by searching for keywords or concepts from the user’s request. If an exact match exists, confirm with the user to reuse it.
2. If it is a recipe-local need, check the corresponding recipe directory for existing configs, datasets, or training workflows that can be used.
3. Once you implement a change, you can start a new subagent with a clean context to do a code review of your own change, and then fix any issues found.
4. If user prefer, run the relevant tests and linters locally. If you need GPU/NPU resources, first check `CUSTOM.md` for any instructions on how to access them, check the local environment, or ask the user for help if you cannot access them.
5. Summarize the change and impacts to the user.

## Build, Test, and Development Commands

- `uv venv --python=3.12 && source .venv/bin/activate`
  - Create and activate the local environment
- `uv sync`
  - Install project dependencies
- `pre-commit install`
  - Install local hooks
- `pre-commit run --all-files`
  - Run lint checks aligned with CI
- `pytest -q`
  - Run tests
- `torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/<experiment>/configs/<config_name>.yaml`
  - Example training launch command

## Testing Rules

- Use `pytest`
- Name test files `test_<feature>.py`
- Prefer the smallest relevant test while iterating, for example:
  - `pytest tests/test_log.py -q`
- Consider broader test coverage before final submission

Recipe-local test rules:

- Put recipe-specific tests inside the corresponding recipe directory
- If a recipe-local test imports `recipes.*`, add a local `conftest.py` under that recipe’s `tests/`
- Do not rely on the top-level `tests/conftest.py` for recipe test discovery

## Git Rules

- Follow the existing commit title style:
  - `feat:`
  - `fix:`
  - `chore:`
  - `enhance:`
- Reference related issues or PRs when applicable
- PRs should include:
  - purpose
  - key changes
  - validation commands
  - config impacts

### Skill Test
- When a task explicitly or implicitly applies a skill, the agent must treat that skill's validation and test requirements as mandatory completion criteria rather than optional guidance.
- Before any skill-related validation work, read the `Recipe-Local Skill Tests` section in `skills/README.md` and follow that workflow before proceeding.
- Skill-required validation must be executed using the exact workflow defined by the skill and the repository's skill-testing conventions, including recipe-local artifacts such as `skill_tests/skill_manifest.yaml`, `skill_tests/<skill-id>/test_spec.yaml`, and the required `structure` / `runtime` / `smoke` layers when the skill calls for them.
- Default to no local preflight before skill validation. Do not add a local pre-check unless it is narrowly scoped, quick, and likely to catch an obvious non-skill issue such as a syntax error, import error, or malformed test file.
- Never run a full local dry run of the skill's `structure`, `runtime`, or `smoke` layer before launching the required subagent validation. In particular, do not locally pre-run the same recipe-local skill test files or launcher path just to "check first".
- If a local preflight is truly necessary, keep it to the minimum needed to unblock the real validation flow, stop after the first useful signal, and do not treat it as part of the reported skill-validation result.
- Run all validation only in fresh subagents with `fork_context=false`. Do not run these `python -m tests.test_skills` commands from the main agent's local terminal, background terminal sessions, or any other non-subagent shell fallback.
- Ordinary unit tests, ad hoc `pytest` runs, or partial smoke checks do not substitute for skill validation unless the skill documentation explicitly allows that substitution.
- A skill-driven task is not complete until the required skill validation has been run and passed, or the environment limitation has been identified precisely and reported together with the exact command the user should run in a real environment.
- When skill validation fails, do not mark the work complete, do not manually override the manifest to a passing state, and do not omit the failing layer from the reported validation result.
- Skill-related validation should prefer the real GPU execution path whenever the skill or recipe meaningfully depends on accelerator, distributed, launcher, logger, checkpoint, or parallel behavior; do not default to CPU-only or single-process `gloo` as a convenience fallback. Do not silently replace a GPU-expected `runtime` or `smoke` test with a weaker CPU variant just to get a local pass. Only use a CPU fallback when the skill documentation explicitly allows it and the CPU path still validates the same required capability.
- If the required GPU or distributed environment is unavailable, do not invent a downgraded local substitute. Report the limitation precisely and print the exact `python -m tests.test_skills ...`, launcher, and any required environment commands the user should run in a real GPU environment instead.

## Commit & Pull Request Guidelines
- Follow existing history style: short, imperative subjects with prefixes like `feat:`, `fix:`, `chore:`, `enhance:`.
- Reference related issues/PRs when applicable (for example `(#9)`).
- PRs should include: purpose, key changes, validation commands run, and config impacts. Add screenshots only for UI/data-viewer changes.
