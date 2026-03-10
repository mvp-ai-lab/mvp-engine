---
name: pr-skill-review
description: Review a PR that modifies skill files (SKILL.md, references under mvp_engine/skills/). Apply the skill review checklist and output concrete issues, suggestions, and what to keep.
---

# Review Skill PR

## When to use

- The user asks you to review a PR that changes files under `mvp_engine/skills/` (e.g. SKILL.md, references/*.md).
- The user asks for a review focused on skill content (pattern, completeness, clarity, fit with skill philosophy, test guidance).

## Required inputs

- PR context (base/head branch, or diff of the PR).
- Optionally: which skill(s) or paths to focus on.

## Workflow

1. **Identify skill changes**
   - List all files in the PR under `mvp_engine/skills/` (any language).
   - If the PR touches both code and skills, scope the review to skill-related files only (or agree with the user to review only skills).

2. **Apply the skill review checklist**
   - Read and apply the checklist in [references/skill-review-checklist.md](references/skill-review-checklist.md) (same dimensions as pr-feedback's feedback-checklist).
   - For each changed skill (or new skill), go through: Accuracy, Completeness, Clarity and consistency, Fit with skill philosophy, Test guidance.
   - Use “Skill location” in the checklist when referring to main workflow, examples, or test templates in comments.

3. **Output for the author**
   - List **concrete issues and suggestions** with file and section references (e.g. `path/to/SKILL.md § When to use`).
   - If something is **good as-is**, say so briefly so the author knows what to keep.
   - Keep feedback actionable: one item per point, with location and suggested change or question.

## Output template

- **Scope**
  - `Files reviewed: <list of skill files>`
- **Issues / suggestions**
  - `file:line or § section | issue or suggestion`
- **Good as-is**
  - Brief note on what to keep.
- **Summary**
  - 1–2 lines: overall fit and main follow-ups.

## Read on demand

- [references/skill-review-checklist.md](references/skill-review-checklist.md): full checklist to apply when **reviewing** skill content (accuracy, completeness, clarity, philosophy, tests).
