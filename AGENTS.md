# Repository Rules (English Agent Version)

This file is the English rules entry for agents working in this repository. Its rules should stay aligned with `AGENTS_zh.md`; if user instructions, system rules, or higher-priority repository rules conflict with this file, follow the higher-priority rules.

## Repository Purpose

This repository contains the core engine, shared utilities, and experiment-specific configurations for training vision models and VLMs. The main goals are:

- Use `mvp_engine/` for stable, generic, reusable training infrastructure
- Use `recipes/` for experiment-specific engines, models, datasets, and configs
- Use `skills/` for structured agent-facing task instructions

Global principles:

- Never over-abstract or over-encapsulate
- Put capabilities that can be expressed as stable generic interfaces in `mvp_engine/`
- Put experiment-specific logic in `recipes/`
- Put recurring workflows that are useful for agents but do not fit a clean reusable API in `skills/`

## Reading Order and High-Priority Rules

Priority reading order:

1. Read `AGENTS.md` files in the current directory and ancestor directories first
2. Read `README.md` when you need project purpose, structure, or usage entrypoints
3. If `CUSTOM.md` exists in the repository, read it as well

High-priority working rules:

- Try not to modify `mvp_engine/`. If you need to, ask the user for confirmation first.
- Check whether the repository already provides the needed capability before writing new code.
- Do not reimplement existing logging, distributed, or training infrastructure.
- If the correct entrypoint, config, workflow, or module is unclear, inspect the repository first and then confirm with the user.

## Repository Structure and Responsibilities

### `mvp_engine/`

- The framework core
- Provides launch entrypoints, config merging, training loops, distributed initialization, logging, and checkpoint infrastructure
- Treat it as stable core by default; do not change it casually

Key points:

- `launch.py` is the unified launch entrypoint
- `config/` provides default configs and schema
- `engine/` provides training base classes
- `distributed/` provides parallel training infrastructure
- `utils/log/` already provides logging output and metric aggregation

### `recipes/`

- Task examples and config entrypoints
- Training-related tasks usually start from `recipes/*/configs/*.yaml`
- Experiment-specific logic should live in `recipes/`

Two especially important example entrypoints:

- `minimal_vlm`
  - A minimal runnable VLM fine-tuning example
- `openbee`
  - A more complete reference for three-stage VLM training

### `skills/`

- Structured task instructions for agents
- `en/` contains English skills, and `zh-cn/` contains Chinese skills
- Reuse existing skills before creating new ones

Parent directory responsibilities:

- `experiment/`
  - Skills for experiment result analysis
- `git/`
  - Skills for Git collaboration, PR checks, and feedback
- `model/`
  - Skills for model migration and MFU / FLOPs utilization
- `parallel/`
  - Skills for FSDP2, TP, and related parallel training work
- `recipe/`
  - Skills for recipe templates and recipe-related workflows
- `skills/`
  - Skills for creating and maintaining skills
- `training/`
  - Skills for training capability enhancements

For detailed inputs, outputs, failure modes, and reuse rules, refer to `skills/AGENTS.md` and each skill's `SKILL.md`.

### `tests/`

- Records validated behavior
- Add tests when behavior changes

Useful existing tests include:

- `tests/test_config_schema.py`
  - Default config and schema compatibility
- `tests/test_launch.py`
  - Launch entrypoint and recipe auto-import behavior
- `tests/test_log.py`
  - Logging and metric aggregation behavior

## Working Rules

Defensive coding: Don't add error handling, fallbacks, or validation **for scenarios that can't happen**. Trust internal code and framework guarantees. **Only validate at system boundaries** (user input, external APIs).

Recommended decision order:

1. After the user states a need, first check whether `skills/` already contains a matching skill
2. If no exact skill exists, check whether a similar skill exists and offer that option to the user for confirmation
3. If `skills/` does not provide a suitable entrypoint, inspect `mvp_engine/`, `recipes/`, or `tests/` for an existing implementation
4. If the repository already contains the needed implementation, prefer reuse, integration, or minimal extension, and confirm the plan with the user
5. Only when neither `skills/` nor existing implementations satisfy the need should you ask the user whether to add new functionality

Stable rules:

- Do not begin by traversing all `recipes/` or all directories; first locate the most relevant skill, module, or config for the user’s request

Boundary rules:

- Try not to modify `mvp_engine/`. If you need to, ask the user for confirmation first.
- Do not put experiment-specific logic into shared framework code.
- Do not reimplement infrastructure that already exists.
- If a matching skill exists in the repository, reuse it first.
- If `mvp_engine/` or `recipes/` already provide the needed capability, prefer integration, reuse, or minimal extension.

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
- `torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/vit_classification/configs/stage1.yaml`
  - Example training launch command

## Code Style and Implementation Constraints

- Python 3.12
- 4-space indentation
- Maximum line length 120
- Use `snake_case` for functions, modules, and files
- Use `PascalCase` for classes
- Use clear config names such as `stage1.yaml` and `stage2.yaml`

Implementation constraints:

- Code in `mvp_engine/` should be generic, reusable, clean, minimal, and easy to maintain
- Never over-abstract or over-encapsulate
- Put experiment-specific logic in `recipes/<experiment>/`
- Keep comments and documentation clear enough to explain intent and behavior
- Avoid deprecated APIs
- Avoid unnecessary helper functions

## Testing Rules

- Use `pytest`
- Name test files `test_<feature>.py`
- Add tests when behavior changes, especially for:
  - engine loops
  - logging
  - distributed behavior
  - dataset handling
- Prefer the smallest relevant test while iterating, for example:
  - `pytest tests/test_log.py -q`
- Consider broader test coverage before final submission

Recipe-local test rules:

- Put recipe-specific tests inside the corresponding recipe directory
- If a recipe-local test imports `recipes.*`, add a local `conftest.py` under that recipe’s `tests/`
- Do not rely on the top-level `tests/conftest.py` for recipe test discovery

## Only When Submitting

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
