# Repository Guidelines

## Project Overview

This repository contains the core training engine and utilities for vision and llm models, along with experiment-specific configurations and code under `recipes/`. It is designed to facilitate scalable training of large models using PyTorch. The aim is to provide a modular, extensible, and easy-to-use framework for researchers and engineers. Never over-abstract or over-encapsulate.

## Project Structure & Module Organization
- `mvp_engine/`: core package.
- `mvp_engine/engine/`: training orchestration and engine base classes.
- `mvp_engine/dataset/`: dataset builders and data pipeline utilities.
- `mvp_engine/distributed/` and `mvp_engine/utils/`: distributed/runtime helpers and other utilities.
- `skills/`: agent skills — structured guides for tasks that have clear patterns but cannot be generalized into a single API (for example gradient checkpointing, FSDP wrap policies). Organized by language (`en/`, `zh-cn/`) and category (`training/`, `parallel/`, `model/`, `data/`, `debug/`, `recipe/`). See `skills/README.md` for overview and `skills/en/README.md` or `skills/zh-cn/README.md` for design rationale.
- `recipes/`: experiment-specific engines, models, datasets, and Hydra YAML configs (for example `recipes/tomatovit/configs/`).
- `tests/`: pytest suite (`test_*.py`) and shared fixtures (`conftest.py`).
- `tools/dataviewer/`: local data viewer app.
- `assets/`, `data/`, `outputs/`, `pretrained/`: static assets, local data links, run artifacts, and model weights.

## Build, Test, and Development Commands
- `uv venv --python=3.12 && source .venv/bin/activate`: create/activate local env.
- `uv sync`: install project dependencies from `pyproject.toml`/`uv.lock`.
- `pre-commit install`: install local hooks.
- `pre-commit run --all-files`: run the same lint checks as CI.
- `pytest -q`: run tests.
- `torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/tomatovit/configs/stage1.yaml`: launch a demo distributed training.

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
- Add or update tests for any behavior change (engine loop, logging, distributed behavior, or dataset handling).
- Prefer targeted runs while iterating (for example `pytest tests/test_log.py -q`) and run full suite before opening a PR.

## Commit & Pull Request Guidelines
- Follow existing history style: short, imperative subjects with prefixes like `feat:`, `fix:`, `chore:`, `enhance:`.
- Reference related issues/PRs when applicable (for example `(#9)`).
- PRs should include: purpose, key changes, validation commands run, and config impacts. Add screenshots only for UI/data-viewer changes.
