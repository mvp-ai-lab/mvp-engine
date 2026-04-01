---
name: pr-gate
description: Use before pushing or opening or updating a PR. Inspect the branch diff, clean up docstrings and typing in touched APIs, run the agreed quality gates, and summarize residual risk.
---

# Pre-PR Quality Gate

## Goal

- Review the current branch before push or PR update.
- Tighten docstrings and type hints in touched public APIs when behavior or signatures changed.
- Run the agreed lint and test gates and report any residual risk.

## Required Inputs

- The base branch, usually `main`.
- The scope to review:
  - full branch (`HEAD` vs `origin/<base>`)
  - last N commits
  - explicit commit list
- Quality-gate commands. Default to `pre-commit run --all-files` and `pytest -q` unless the repo needs something narrower.

## Workflow

### 1. Sync the baseline

- Update the local baseline for the chosen base branch.
- Return to the working branch and confirm the workspace state before reviewing diffs.

### 2. Build change context

- Inspect the commit graph for the scoped range.
- Build a file-level change map.
- Read the critical diffs instead of relying only on filenames.

### 3. Clean up docstrings and typing in touched code

- Limit edits to functions, classes, and modules that the branch already touched.
- Apply these rules:
  - new public functions and classes need docstrings
  - new or changed public functions should have explicit parameter and return type hints where the language supports them
  - when signatures, returns, side effects, or behavior change, update docstrings and type hints together
  - tighten stale or inaccurate types in touched code instead of leaving broad `Any` where a concrete type is available
  - private trivial helpers may skip docstrings
- Keep documentation aligned with actual behavior and avoid filler text.

### 4. Run the quality gates

- Run formatting and lint checks first, then tests.
- Fix failures related to the current change set before reporting out.
- If the full gate is too expensive, run the strongest targeted checks you can justify and state the missing coverage.

### 5. Prepare the PR-ready summary

- Rank findings by severity.
- Record the commands that ran and the outcome of each.
- Suggest a short commit message if the user asked for one.

## Validation

- The review scope matches the requested base branch and commit range.
- Changed public APIs have docstrings and type hints aligned with their real behavior.
- Quality-gate commands were run or explicitly deferred with a clear reason.
- Residual risks are stated only for areas that were not fully validated.

## Output

- Findings:
  - `severity | file:line | issue | recommendation`
- Validation:
  - `command | result`
- Residual Risks:
  - `not validated yet`
- Suggested Commit Message:
  - short imperative summary when useful

## Read On Demand

- Read [references/docstring-and-typing.md](references/docstring-and-typing.md) when touched code needs docstring and type-hint cleanup guidance.
