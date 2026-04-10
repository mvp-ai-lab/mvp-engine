---
name: pr-skill-review
description: Review a PR that changes files under skills/. Apply the skill review checklist, identify concrete issues and strengths, and keep the feedback tied to specific files and sections.
---

# Review Skill PR

## Goal

- Review only the skill-related portion of a PR.
- Apply the repository's skill review checklist consistently.
- Produce feedback that is concrete enough for the author to act on immediately.

## Required Inputs

- PR context such as base/head branches or a PR diff.
- Optional focus paths or skill names if the user wants a narrower review.

## Workflow

### 1. Scope the skill changes

- List all changed files under `skills/`.
- If the PR also changes code outside `skills/`, keep the review scoped to skill files unless the user asks for a broader review.

### 2. Apply the skill review checklist

- Read [references/skill-review-checklist.md](references/skill-review-checklist.md).
- Review each changed or newly added skill against these dimensions:
  - accuracy
  - completeness
  - clarity and consistency
  - fit with skill philosophy
  - test or validation guidance
- Use concrete section references when discussing workflow, examples, or validation guidance.

### 3. Prepare author-facing review feedback

- List concrete issues and suggestions with file and section references.
- Call out content that is already strong so the author knows what to keep.
- Keep each review point actionable: one issue or suggestion per item, with a recommended change or a must-ask question.

## Validation

- Every changed skill file under review was inspected.
- Findings reference a concrete file location or section.
- The review separates problems to fix from content that is already good as-is.
- The checklist dimensions were actually applied instead of giving generic style feedback.

## Output

- Scope:
  - `Files reviewed: <list of skill files>`
- Issues / Suggestions:
  - `file:line or section | issue or suggestion`
- Good As-Is:
  - short notes on content worth keeping
- Summary:
  - 1-2 lines on overall fit and next follow-ups

## Read On Demand

- Read [references/skill-review-checklist.md](references/skill-review-checklist.md) before reviewing skill content.
