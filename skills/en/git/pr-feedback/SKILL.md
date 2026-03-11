---
name: pr-feedback
description: Use after a PR is open and review comments arrive. Covers comment triage, targeted fixes, regression checks, and reviewer response drafting.
---

# Handle PR Feedback

## When to use

- The user asks to handle reviewer comments on an open PR.
- The user asks to batch-fix review feedback and prepare responses.

## Required inputs

- PR context (base/head branch or equivalent diff range).
- Reviewer comments (inline comments, summary comments, or issue links).
- Validation commands required before re-push.

## Workflow

1. Collect review context
- Gather unresolved comments and categorize by severity/type.
- Map each comment to concrete files/lines/commits.

2. Triage and plan
- Group by action type:
  - must-fix correctness issues
  - design/readability improvements
  - clarification-only responses
- Identify conflicts or ambiguous comments to clarify with user.

3. Implement targeted fixes
- Keep each fix scoped to comment intent.
- Update docstrings and type hints/comments when behavior contract changed.
- Avoid unrelated cleanup in the same patch.

4. Re-validate
- Run required lint/test commands.
- If full validation is expensive, run targeted tests and report gaps.

5. Draft reviewer responses
- For each comment, provide:
  - what changed
  - where it changed (`file:line`)
  - validation evidence
- If not changing code, provide concise technical rationale.

## Output template

- Comment Resolution
  - `comment id | action (fixed/clarified/pending) | file:line`
- Validation
  - `command | result`
- Pending Items
  - `needs user decision / reviewer confirmation`

## Read on demand

- [references/feedback-checklist.md](references/feedback-checklist.md): when the PR is about a skill, use this to ensure replies address the same dimensions reviewers use.
- [../references/docstring-and-typing.md](../references/docstring-and-typing.md): docstring and typing update rules when behavior changed.
