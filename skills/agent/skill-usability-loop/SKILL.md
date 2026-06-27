---
name: skill-usability-loop
description: Optimize a repository skill for a first-time user by running a
  fixed six-role fresh-subagent loop until a fresh novice can produce correct
  validated output within the declared task scope. Use when a user provides a
  target skill, novice task, optional evaluation-only demo/reference, and round
  budget; the novice coder must implement only from the task and skill
  instructions, never from a user-provided reference implementation. Derive
  per-round reviewer/tester checks and final reporting from the target skill
  instead of asking the user to script the loop.
---

# Skill Usability Loop

## Goal

Improve a skill so a first-time agent can use it successfully and produce a
correct result within the declared task and validation scope, without turning
the skill into a demo-recipe bug log.

Use this skill when the user asks to optimize, audit, or regression-test the
first-use experience of a skill.

## User Prompt Contract

Expect the user to provide only task-level inputs:

- target skill path;
- novice task;
- optional demo/reference code area, used only as evaluation or diagnostic
  material and never as novice-coder implementation context;
- optional round budget or correctness goal.

Do not require the user to write per-round instructions, reviewer checklists,
tester commands, failure taxonomy, or final output format. Derive those from
this loop plus the target skill's `Required Inputs`, `Workflow`, `Validation`,
and `Output` sections. Ask the user only when the target skill and repository
context do not reveal the task, correctness standard, or runtime resources.

## Required Inputs

Identify these before starting:

- target skill path, such as `skills/parallel/context-parallel/SKILL.md`;
- realistic one-sentence user task for the novice run;
- optional demo recipe or reference implementation used only as a usability
  probe, oracle, or comparison artifact for reviewer/tester/diagnoser roles;
- correctness standard for the novice output, derived first from the target
  skill's validation and output requirements, including required static,
  runtime, parity, or reviewer checks;
- stop criteria, usually two fresh novice runs with no unresolved correctness
  findings in the declared scope, or a fixed round budget;
- whether runtime validation is required in this round and the exact resources
  or scheduler parameters needed to run it.

If the target skill or task is unclear, ask before spawning subagents. If only
the reviewer/tester checklist is unclear, read the target skill and derive it;
do not ask the user to spell out loop mechanics.

## Subagent Roles

Use exactly the six project-scoped Codex custom agents in `.codex/agents/`:

1. `.codex/agents/novice-coder.toml`
2. `.codex/agents/code-reviewer.toml`
3. `.codex/agents/tester.toml`
4. `.codex/agents/skill-diagnoser.toml`
5. `.codex/agents/skill-writer.toml`
6. `.codex/agents/writing-supervisor.toml`

Every subagent must be fresh with `fork_context=false`. Close each subagent
after its report is captured. Do not store subagent prompt files under this
skill; `.codex/agents/*.toml` is the prompt source of truth.

## Workflow

### Round Worktree Isolation

For every round that may produce code changes, create a fresh temporary git
worktree before spawning the novice. The worktree must contain the intended
baseline code plus the current target skill revision for that round.

All novice code edits for that round must happen only inside that worktree; do
not let probe code changes modify or dirty the main repository checkout. Pass
the same worktree path to `code-reviewer` and `tester`, and require them to
inspect and validate the novice output from that worktree. At the end of the
round, after all reports are captured, remove the temporary worktree.

Use the main repository checkout's existing `uv` virtual environment for all
code execution, testing, and validation from the temporary worktree. Do not
create or sync a separate virtual environment inside the round worktree unless
the user explicitly asks.

### 1. Freeze The Round

Record:

- target skill revision and changed reference files;
- temporary git worktree path used for this round, if code output will be
  produced;
- one-sentence novice task;
- demo recipe or reference implementation, if any, and which non-novice roles
  may use it for evaluation;
- derived correctness checklist from the target skill;
- validation budget and unavailable resources.

Do not change the skill during the novice run.

Do not include user-provided reference implementations, completed patches,
expected diffs, or solution walkthroughs in the novice-coder prompt or
workspace. The novice-coder probe measures whether the skill itself is
sufficient for a fresh implementation.

If runtime validation is required and resources are available, use them. If
resources are unavailable, record the exact missing values or credentials before
starting probes so blocked validation is not mistaken for success.

### 2. Novice Implementation Probe

Spawn `novice-coder` with only the one-sentence user task and any explicit skill
name/path needed to trigger the skill. Do not pass any user-provided reference
implementation, solution patch, reviewer oracle, or demo code that reveals the
intended implementation. If the repository already contains similar code, the
novice may discover it naturally through the task and skill, but the parent
agent must not point the novice at reference material.

The novice may edit code only in the round's temporary git worktree. Its final
report must list:

- files changed;
- validation attempted;
- decisions it inferred from the skill;
- places where it was uncertain.

### 3. Correctness Review

Spawn `code-reviewer` to review the novice output in the same temporary git
worktree.

Pass the reviewer the derived correctness checklist, not a user-authored
checklist. The reviewer should still classify any additional correctness issue
it discovers.

The reviewer must separate findings into:

- implementation bug;
- skill-caused ambiguity;
- bad or misleading example;
- missing validation;
- not a skill issue.

Do not stop only because the reviewer labels a finding as an implementation
bug. A first-time user's implementation bug is usability evidence unless it is
outside the task scope, depends only on unavailable external resources, or is
caused by unrelated repository state.

### 4. Runtime Or Test Probe

Spawn `tester` only on the novice output in the same temporary git worktree,
not on skill edits.

Pass the tester the derived validation checks and ask it to run the smallest
available commands that can expose whether the skill led to a runnable outcome.
If hardware or credentials are unavailable, it reports the exact blocked command
and expected environment.

When the expected correct output is code, the tester must include a validation
that would have failed for each reviewer correctness finding when such a check
can run without unavailable external resources.

### 5. Diagnose Skill Usability

Spawn `skill-diagnoser` with the novice report, code review, and tester report.

The diagnoser must produce a failure taxonomy:

- `missing_rule`
- `ambiguous_wording`
- `misleading_example`
- `rule_too_buried`
- `validation_unclear`
- `over_specific_to_demo`
- `not_skill_issue`

Only `missing_rule`, `ambiguous_wording`, `misleading_example`,
`rule_too_buried`, `validation_unclear`, and `over_specific_to_demo` may drive
skill edits.

Classification rules:

- If the novice missed a rule that is present but did not affect the output,
  classify it as `rule_too_buried`, `ambiguous_wording`, or
  `validation_unclear`, not `not_skill_issue`.
- If a correctness issue could have been caught by a cheap static, config, or
  unit check that the skill did not require, classify it as
  `validation_unclear`.
- Use `not_skill_issue` only for issues outside the task scope, issues caused
  by unrelated dirty state, or checks blocked solely by missing external
  resources after the blocked command is recorded.

### 6. Edit The Skill

Spawn `skill-writer` to make the smallest general-first edits.

Rules for edits:

- keep `SKILL.md` as a short route map;
- put model-, recipe-, or framework-specific detail in `references/`;
- avoid recording round history in `SKILL.md`;
- turn demo failures into general invariants;
- turn repeated novice implementation misses into explicit required actions or
  executable validation checks;
- make copyable snippets real APIs or clearly mark them as pseudocode;
- prefer deleting or moving text over expanding the main skill.

### 7. Writing Review

Spawn `writing-supervisor` to review only the skill/config changes.

The supervisor checks:

- the guidance is general, not demo-recipe-specific;
- first-time readers can find the next action quickly;
- examples are accurate and scoped;
- validation is executable;
- references are read-on-demand rather than required up front.

If it fails, return once to `skill-writer` with only the supervisor findings.
Do not restart the novice implementation until writing passes.

### 8. Regression Probe

After writing passes, start a new round from step 1 with a fresh `novice-coder`.

Do not reuse the previous novice context. Reuse the same one-sentence task
unless intentionally testing a new user path.

If code output is the evaluation artifact, run the regression probe from a clean
temporary git worktree that contains only the updated skill changes and the
intended baseline code. Do not let previous novice edits contaminate the next
run, and remove each round worktree before starting another round.

If the user set a round budget, stop after the last round and report unresolved
correctness findings. Otherwise, stop only after two fresh novice runs produce
no unresolved reviewer or tester correctness findings within the declared
validation scope. Do not stop merely because a remaining issue is now labeled an
implementation bug if it is still a first-time output correctness failure.

If the current round does not meet the stop criteria or stop rationale, continue
the complete loop after the fresh `novice-coder` finishes its new code output:
run correctness review, runtime or test probe, skill diagnosis, skill edit, and
writing review before starting any later regression probe.

## Output

Return:

- Subagent Reports: one line per role with pass/fail and artifact links;
- Failure Taxonomy: grouped by skill-caused vs not skill-caused;
- Skill Changes: files changed and general principle behind each edit;
- Regression Result: whether the same skill-caused failure recurred;
- Correctness Result: whether fresh novice output has unresolved reviewer or
  tester correctness findings within scope;
- Next Round Recommendation: stop, run another novice probe, or escalate to
  code/kit changes.

## Read On Demand

- `../../../.codex/agents/*.toml`: canonical persisted Codex custom agent
  definitions.
