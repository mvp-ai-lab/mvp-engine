# Base Template

Use this template when creating a new skill from scratch. Replace placeholders, then trim any section that is not needed.

```md
---
name: <skill-name>
description: <What this skill does. When it should be used.>
---

# <Skill Title>

## Goal

- State the goal.
- State the boundaries.
- State the expected deliverable.

## Required Inputs

- List the required context.
- List the required files, configs, or constraints.

## Workflow

### 1. Gather Context

- Explain what to inspect first.
- Explain what decisions depend on that context.

### 2. Make Changes

- Explain the main implementation path.
- Explain any routing rules for common variants.

## Validation

- List the checks or commands required to confirm the result.

## Output

- State what the final response must include.

## Read On Demand

- Point to any optional `references/` or `scripts/` entries and when to use them.
```
