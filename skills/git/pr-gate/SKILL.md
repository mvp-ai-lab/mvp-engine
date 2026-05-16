---
name: pr-gate
description: Run a pre-PR or pre-push quality gate by inspecting the branch diff,
  tightening touched public API docstrings/types, running validation, and
  summarizing residual risk.
---

# PR Gate

## Goal

Prepare a branch for PR or push:

- inspect the requested diff scope;
- fix docstrings and type hints in touched public APIs when needed;
- run agreed lint/test gates;
- report findings, validation, and residual risk clearly.

## Required Inputs

Identify these before editing:

- base branch or diff range;
- scope: full branch, last N commits, explicit commits, or changed paths;
- quality-gate commands;
- whether the working tree has unrelated local edits.

Default validation is `pre-commit run --all-files` and `pytest -q` only when the
repo and environment make those reasonable. Otherwise use the strongest targeted
checks and explain the scope.

## Workflow

### 1. Establish Diff Scope

Check branch and workspace state, then inspect the chosen range:

```bash
git branch --show-current
git status --short
git diff --name-status <base>...HEAD
```

Do not overwrite unrelated dirty files.

### 2. Review Touched Code

Read critical diffs, not only filenames. Focus on public API changes, runtime
behavior, config contracts, tests, docs, and generated artifacts.

### 3. Tighten Docstrings And Types

Read `references/docstring-and-typing.md` when touched APIs need cleanup.

Only edit files already in scope. Keep docstrings and type hints aligned with
actual behavior; avoid filler comments and broad `Any` unless justified.

### 4. Run Quality Gates

Run formatting/lint first, then tests. Fix failures caused by the current change
set. If a gate is blocked, record the blocker and residual risk.

### 5. Prepare PR-Ready Summary

List findings by severity, validation commands and outcomes, and any remaining
unvalidated paths.

## Output

- Findings: `severity | file:line | issue | recommendation`.
- Changes Made: docstring/type/test/docs cleanup applied.
- Validation: `command | result`.
- Residual Risks: unvalidated paths.
- Suggested Commit Message: include only when useful or requested.

## Read On Demand

- `references/docstring-and-typing.md`: cleanup scope and examples.
