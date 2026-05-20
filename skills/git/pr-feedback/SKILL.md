---
name: pr-feedback
description: Triage reviewer feedback on an open PR, implement scoped fixes,
  rerun validation, and prepare reviewer-ready responses tied to comments,
  changed files, and validation evidence.
---

# PR Feedback

## Goal

Resolve reviewer feedback without broadening the PR:

- account for every unresolved review comment;
- implement only changes that map to reviewer intent;
- preserve unrelated user edits;
- rerun the strongest relevant validation;
- draft concise responses that reviewers can verify.

## Required Inputs

Identify these before editing:

- PR URL/number or equivalent base/head diff;
- unresolved inline and top-level reviewer comments;
- base branch and current working branch;
- validation commands expected for the PR;
- whether any local uncommitted changes are unrelated.

Ask the user only when reviewer intent or permission to mutate/push is unclear.

## Workflow

### 1. Collect Review Context

Gather comments and map each to:

- file and line or PR-level topic;
- severity: correctness, design, readability, docs, tests, or clarification;
- action: fix, explain, defer, or ask for clarification.

### 2. Plan Scoped Changes

Group comments that require the same edit. Keep explanation-only comments
separate from code changes.

If comments conflict, identify the conflict before editing and ask for a
decision only when the repo context cannot resolve it.

### 3. Implement Fixes

Make the smallest correct change for each actionable comment. Update docstrings,
typing, tests, or docs only when the behavior contract changed or the reviewer
asked for it.

Avoid unrelated cleanup and do not revert local changes you did not make.

### 4. Re-Validate

Run targeted tests for touched behavior first, then broader gates when feasible.
If full validation is too expensive or blocked, state the exact remaining gap.

### 5. Draft Responses

For every comment, prepare:

- status: fixed, clarified, deferred, or pending;
- what changed or why no code change was needed;
- changed file/line references;
- validation command and result.

Read `references/feedback-checklist.md` when the PR changes skills.

## Output

- Comment Resolution: `comment id/topic | fixed/clarified/deferred/pending`.
- Changes: files changed and why.
- Validation: `command | result`.
- Reviewer Responses: concise text ready to post.
- Pending Items: decisions or reviewer confirmations still needed.

## Read On Demand

- `references/feedback-checklist.md`: skill-PR feedback dimensions.
- `../pr-gate/references/docstring-and-typing.md`: docstring/type cleanup rules.
