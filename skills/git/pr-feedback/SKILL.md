---
name: pr-feedback
description: Use after a PR is open and review comments arrive. Triage reviewer comments, implement scoped fixes, re-run validation, and draft reviewer-ready responses with file references.
---

# Handle PR Feedback

## Goal

- Resolve open reviewer feedback on an existing PR with targeted changes.
- Keep code changes scoped to comment intent and make validation evidence easy to report.
- Produce reviewer responses that are ready to post.

## Required Inputs

- PR context such as base/head branches or an equivalent diff range.
- Reviewer comments, including inline comments, summary comments, or linked issues.
- Validation commands that should run before the branch is pushed again.

## Workflow

### 1. Collect review context

- Gather all unresolved comments and map each one to concrete files, lines, or commits.
- Note whether the comment is blocking correctness, design/readability, or clarification-only.

### 2. Triage and plan

- Group comments by action type:
  - must-fix correctness issues
  - design or readability improvements
  - explanation-only responses
- Mark conflicts or ambiguous comments that still need user or reviewer clarification.

### 3. Implement targeted fixes

- Keep each code change tightly aligned with the comment that motivated it.
- Update docstrings, type hints, or comments when the behavior contract changes.
- Avoid unrelated cleanup in the same patch.

### 4. Re-validate

- Run the required lint and test commands.
- If full validation is too expensive, run the targeted checks that cover the changed paths and report the remaining gap explicitly.

### 5. Draft reviewer responses

- For each comment, state:
  - what changed
  - where it changed using `file:line`
  - what validation supports the change
- If code is not being changed, give a concise technical rationale instead of a vague refusal.

## Validation

- Every unresolved comment is accounted for as fixed, clarified, or still pending.
- Each code change maps back to a concrete reviewer comment.
- Validation results are recorded, or any remaining validation gap is called out explicitly.
- Reviewer responses cite the changed locations precisely enough for the reviewer to follow them.

## Output

- Comment Resolution:
  - `comment id | action (fixed/clarified/pending) | file:line`
- Validation:
  - `command | result`
- Pending Items:
  - `needs user decision / reviewer confirmation`

## Read On Demand

- Read [references/feedback-checklist.md](references/feedback-checklist.md) when the PR changes skills and the response should match the same review dimensions used by skill reviewers.
- Read [../pr-gate/references/docstring-and-typing.md](../pr-gate/references/docstring-and-typing.md) when behavior changed and the touched code needs docstring or typing cleanup.
