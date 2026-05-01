# Repository Guidelines

## Project Overview

This repository contains the core training engine and utilities for vision and llm models, along with experiment-specific configurations and code under `recipes/`. It is designed to facilitate scalable training of large models using PyTorch. The aim is to provide a modular, extensible, and easy-to-use framework for researchers and engineers. Never over-abstract or over-encapsulate.

## Project Structure & Module Organization
- `mvp_engine/`: core package.
- `mvp_engine/engine/`: training orchestration and engine base classes.
- `mvp_engine/dataset/`: dataset builders and data pipeline utilities.
- `mvp_engine/distributed/` and `mvp_engine/utils/`: distributed/runtime helpers and other utilities.
- `skills/`: agent skills — structured guides for tasks that have clear patterns but cannot be generalized into a single API (for example gradient checkpointing, FSDP wrap policies). Organized by language (`en/`, `zh-cn/`) and category (`training/`, `parallel/`, `model/`, `data/`, `debug/`, `recipe/`). See `skills/README.md` for overview and `skills/en/README.md` or `skills/zh-cn/README.md` for design rationale.
- `recipes/`: experiment-specific engines, models, datasets, and Hydra YAML configs (for example `recipes/vit_classification/configs/`).
- `tests/`: pytest suite (`test_*.py`) and shared fixtures (`conftest.py`).
- `tools/dataviewer/`: local data viewer app.
- `assets/`, `data/`, `outputs/`, `pretrained/`: static assets, local data links, run artifacts, and model weights.

## Highest Priority User Defined Custom Rules and Information

- Read `CUSTOM.md` for custom rules and information defined by users. This file contains important guidelines and best practices that are specific to this project and may not be covered in general coding standards. It is essential to review this document to ensure that your contributions align with the user's requirements and expectations.

## Build, Test, and Development Commands
- `uv venv --python=3.12 && source .venv/bin/activate`: create/activate local env.
- `uv sync`: install project dependencies from `pyproject.toml`/`uv.lock`.
- `pre-commit install`: install local hooks.
- `pre-commit run --all-files`: run the same lint checks as CI.
- `pytest -q`: run tests.
- `torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/vit_classification/configs/stage1.yaml`: launch a demo distributed training.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indentation, max line length 120.
- Formatting/linting is enforced through pre-commit: `ruff-check --fix`, `ruff-format`, and `isort --profile black`.
- Use `snake_case` for functions/modules/files, `PascalCase` for classes, and clear config names (for example `stage1.yaml`, `stage2.yaml`).
- The code in `mvp_engine/` should be generic, reusable, clean, minimal, well-documented, and easy-to-maintain.
- Keep experiment-specific logic under `recipes/<experiment>/` rather than in shared engine code.
- NEVER over-abstract or over-encapsulate.
- Your code should have clear comments and documentation to help others understand your thought process and the functionality of your code.
- Avoid using deprecated functions, APIs, and libraries, and always keep your dependencies up to date.

## Testing Guidelines
- Use `pytest` and place tests in `tests/` as `test_<feature>.py`.
- Add or update tests for any behavior change (engine loop, logging, distributed behavior, or dataset handling). Only keep test files that are very important.
- Prefer targeted runs while iterating (for example `pytest tests/test_log.py -q`) and run full suite before opening a PR.
- For recipe-specific code under `recipes/`, add tests in the same directory (for example `recipes/vit_classification/tests/test_*.py`).
- If a recipe-local test imports `recipes.*`, add a local `conftest.py` in that recipe's `tests/` directory to insert the repository root into `sys.path`; do not rely on the top-level `tests/conftest.py` for recipe test discovery.

### Skill Test
- When a task explicitly or implicitly applies a skill, the agent must treat that skill's validation and test requirements as mandatory completion criteria rather than optional guidance.
- Before any skill-related validation work, read the `Recipe-Local Skill Tests` section in `skills/en/README.md` and follow that workflow before proceeding.
- Skill-required validation must be executed using the exact workflow defined by the skill and the repository's skill-testing conventions, including recipe-local artifacts such as `skill_tests/skill_manifest.yaml`, `skill_tests/<skill-id>/test_spec.yaml`, and the required `structure` / `runtime` / `smoke` / optional `effectiveness` layers when the skill calls for them.
- Default to no local preflight before skill validation. Do not add a local pre-check unless it is narrowly scoped, quick, and likely to catch an obvious non-skill issue such as a syntax error, import error, or malformed test file.
- Never run a full local dry run of the skill's `structure`, `runtime`, `smoke`, or `effectiveness` layer before launching the required subagent validation. In particular, do not locally pre-run the same recipe-local skill test files or launcher path just to "check first".
- If a local preflight is truly necessary, keep it to the minimum needed to unblock the real validation flow, stop after the first useful signal, and do not treat it as part of the reported skill-validation result.
- Run all validation only in fresh subagents with `fork_context=false`. Do not run these `python -m tests.test_skills` commands from the main agent's local terminal, background terminal sessions, or any other non-subagent shell fallback.
- Ordinary unit tests, ad hoc `pytest` runs, or partial smoke checks do not substitute for skill validation unless the skill documentation explicitly allows that substitution.
- A skill-driven task is not complete until the required skill validation has been run and passed, or the environment limitation has been identified precisely and reported together with the exact command the user should run in a real environment.
- When skill validation fails, do not mark the work complete, do not manually override the manifest to a passing state, and do not omit the failing layer from the reported validation result.
- If a `SKILL.md` declares an effectiveness layer, `test_effectiveness.py` is mandatory for that recipe-local application and all four layers must pass before the skill can be marked `applied`. If the skill does not declare effectiveness, record `last_validated.effectiveness: not_applicable` in the recipe manifest.
- Generated recipe-local `test_spec.yaml` files must explicitly set `requires.effectiveness: true` or `false` according to whether the source `SKILL.md` references `test_effectiveness.py`; do not omit the field.
- Skill-related validation should prefer the real GPU execution path whenever the skill or recipe meaningfully depends on accelerator, distributed, launcher, logger, checkpoint, or parallel behavior; do not default to CPU-only or single-process `gloo` as a convenience fallback. Do not silently replace a GPU-expected `runtime` or `smoke` test with a weaker CPU variant just to get a local pass. Only use a CPU fallback when the skill documentation explicitly allows it and the CPU path still validates the same required capability.
- If the required GPU or distributed environment is unavailable, do not invent a downgraded local substitute. Report the limitation precisely and print the exact `python -m tests.test_skills ...`, launcher, and any required environment commands the user should run in a real GPU environment instead.

## Commit & Pull Request Guidelines
- Follow existing history style: short, imperative subjects with prefixes like `feat:`, `fix:`, `chore:`, `enhance:`.
- Reference related issues/PRs when applicable (for example `(#9)`).
- PRs should include: purpose, key changes, validation commands run, and config impacts. Add screenshots only for UI/data-viewer changes.
