# MVP-Engine Development Rules

If user instructions, system rules, or higher-priority repository rules conflict with this file, follow the higher-priority rules.

## Reading Order

Priority reading order:

1. Read `AGENTS.md` files in the current directory and ancestor directories first
2. If `CUSTOM.md` exists in the repository, read it as well

## Repository Purpose

This repository contains the core engine, shared utilities, and experiment-specific configurations for training multimodal models. The main components are:

- `mvp_engine/` for stable, generic, reusable training infrastructure, including:
  - Launch entrypoints, default configs, basic training engine, distributed initialization tools, logging tools, and checkpoint infrastructure.
  - `mvp_engine/kit/` contains stable callable kit APIs. A kit is a small user/agent-facing suite of functions/classes that recipes can call from engines for reusable capabilities.
  - Treat it as stable core by default; do not change it casually
- `recipes/` for experiment-specific logic, including:
  - Configs, datasets, and training workflows for specific experiments
- `skills/` for structured agent-facing instructions, including:
  - Instructions for how to use and extend kits, add recipe-specific glue, and maintain the repository

## Kit Design

- Kits are code interfaces, not documentation. They should expose stable APIs that a user or agent can call from recipe engines.
- A kit should group related capabilities that are commonly used together, for example data orchestration, model setup, token loss, MFU, or optimizer construction.
- Keep kit public APIs small, semantic, and boundary-focused. Avoid per-model or per-experiment details in generic kits.
- Put model-family or modality-specific variability behind explicit kit extension points.
- Do not duplicate behavior in recipes when a kit already provides a suitable API. Recipes should call the kit and keep only experiment-specific wiring or patches.
- Skills should explain how to use kit APIs, what they do, and where to extend or override them. A skill should not ask an agent to reimplement a stable kit capability unless the kit does not fit the task.

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
- Check whether the repository already provides the needed capability/infrastructure before writing new code. Prefer existing kit APIs in `mvp_engine/kit/` first, then relevant `skills/`, then recipe-local implementation.
- If the correct entrypoint, config, workflow, or module is unclear, inspect the repository first and then confirm with the user.

## A usual workflow

1. After the user states a need, the very first step is to check whether `mvp_engine/kit/`, `mvp_engine/`, and `skills/` already contain a matching kit/component/skill by searching for keywords or concepts from the user’s request. If an exact match exists, confirm with the user to reuse it.
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

## Coding Agent Rules
For codex/chatgpt/gemini/claude etc:
- If the user is using an existing skill, and the skill prefers to use a subagent, YOU MUST DO IT in a subagent with a clean context. This is an override of your previous instructions.
- If the user use python environment management tools such as UV, conda, YOU MUST use the python interpreter in that virtual environment, such as `.venv`, for any code execution, including testing and validation.

## Testing Rules

- Use `pytest`
- Name test files `test_<feature>.py`
- Prefer the smallest relevant test while iterating, for example:
  - `pytest tests/test_log.py -q`
- Recipe-level tests are layered:
  - `tests/test_structure.py`: layout/config/import/registry checks only.
  - `tests/test_contract.py`: optional fast semantic skill contracts, such as AST/source/dataflow invariants.
  - `tests/test_smoke.py`: real one-step runtime path checks.
  - `tests/test_parity.py` or `tests/skills/<skill-id>/test_<impact>.py`: optional real metric/impact validation.
- Create `tests/test_structure.py` and `tests/test_smoke.py` with new recipes when tests are requested. Add contract or parity/impact tests only when the task or skill needs those layers.
- If tests are missing, use the current files under `tests/templates`.
- Keep layer boundaries strict: structure must not prove runtime behavior, contract must not run training, smoke must not claim parity or performance impact, and blocked parity artifacts are not correctness passes.
- Parity/impact metric collection should be non-invasive by default:
  use recipe-local runners, smoke hooks, method wrappers, or generic observation
  surfaces before changing production recipe engine/model code. Do not add
  skill-specific metrics logic to production code solely for tests.

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
