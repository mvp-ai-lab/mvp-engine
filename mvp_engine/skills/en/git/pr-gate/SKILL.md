---
name: pr-gate
description: Use before pushing or opening/updating a PR. Covers commit-level diff inspection, docstring completion for changed APIs, and pre-PR lint/test quality gates.
---

# Pre-PR Quality Gate

## When to use

- The user asks for pre-push cleanup before opening/updating a PR.
- The user asks to add/update docstrings based on current commits.

## Required inputs

- Base branch (default: `main`).
- Scope:
  - full branch (`HEAD` vs `origin/<base>`)
  - last N commits
  - explicit commit list
- Quality-gate commands (default: `pre-commit run --all-files` and `pytest -q`, adjust per repo).

## Workflow

1. Sync baseline
- `git checkout <base>`
- `git pull --ff-only`
- Switch back to the working branch and confirm workspace state.

2. Build change context
- Commit graph: `git log --oneline --decorate --graph <base>..HEAD`
- File-level map: `git diff --name-status origin/<base>...HEAD`
- Inspect critical diffs.

3. Docstring completion from changed commits
- Limit edits to touched functions/classes/modules.
- Rules:
  - New public functions/classes must have docstrings.
  - Update docstrings when signature/returns/side effects/behavior changes.
  - Private trivial helpers may skip docstrings.
- Avoid filler text; describe behavior, IO contract, constraints.

4. Pre-PR quality gates
- Run formatting/lint checks first, then tests.
- Fix failures related to current change set first.
- List not-yet-validated areas.

5. Output format
- Findings ranked by severity.
- Commands run and result summary.
- Suggested commit message.

## Review output template

- Findings
  - `severity | file:line | issue | recommendation`
- Validation
  - `command | result`
- Residual Risks
  - `not validated yet`

## Read on demand

- [references/review-checklist.md](references/review-checklist.md): concise review checklist.
- [references/docstring-rules.md](references/docstring-rules.md): docstring rules and examples.
