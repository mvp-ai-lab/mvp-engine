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

- When a user asks to execute, apply, validate, test, or run a skill for a recipe,
treat that request as an explicit request to use the fresh subagents required by
the repository skill-validation workflow.

- Validation is part of applying the skill and is a mandatory completion criterion,
not optional guidance. Before validation, read `skills/AGENTS.md` and the target
`SKILL.md`, add the recipe-local `skill_tests/` artifacts required by that skill,
and validate through:
  - `python -m tests.test_skills --recipe <recipe> --skill <skill-id>`

- Run these validation commands only in fresh subagents with `fork_context=false`.
Do not run them from the main agent's local terminal, background terminal
sessions, or any other non-subagent shell fallback.

- If `skills/AGENTS.md`, the target `SKILL.md`, or recipe-local test artifacts
define layers such as `structure`, `smoke`, or `effectiveness`, run
each required layer in a separate fresh subagent, in order, stopping on first
failure.

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
