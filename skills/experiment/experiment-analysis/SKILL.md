---
name: experiment-analysis
description: Analyze mvp-engine experiment outputs and produce a structured
  Markdown report grounded in run artifacts, logs, checkpoints, optional eval
  files, and baseline comparisons.
---

# Experiment Analysis

## Goal

Produce one evidence-grounded Markdown report that helps decide what to do next:

- summarize run setup, training dynamics, checkpoints, and failures;
- compare runs only when metrics are comparable;
- use repo-native artifacts as primary evidence;
- use external eval files as optional additional evidence;
- clearly separate direct evidence from inference.

## Required Inputs

Identify these before writing:

- one or more `outputs/<run_id>/` directories;
- `config_<run_id>.yaml`, `log_<run_id>.log`, and `checkpoints/` when present;
- optional `results*.json`, `metrics*.json`, `samples*.jsonl`,
  `predictions*.jsonl`, or `predictions*.csv`;
- optional baseline or reference runs;
- target report path when the user wants an existing report updated.

Ask the user only when the run path, comparison target, or report destination is
ambiguous.

## Workflow

### 1. Discover Artifacts

Start from repo-native outputs:

```bash
rg --files outputs -g 'config_*.yaml' -g 'log_*.log'
find outputs -path '*/checkpoints/*' -maxdepth 3 -type d
```

Read `references/mvp-engine-artifacts.md` when artifact layout or discovery is
unclear.

### 2. Extract Deterministic Run Summaries

Run the bundled helper for each run directory:

```bash
python3 skills/experiment/experiment-analysis/scripts/summarize_run.py \
  --run-dir outputs/<run_id>
```

Use the helper output as notes. Read raw logs directly when warnings, errors,
metric anomalies, or message context matter.

### 3. Analyze Evidence

For each run, collect:

- config metadata: recipe, engine, model, loop policy, total steps, git info;
- latest and best logged metrics;
- warnings, errors, stalls, regressions, and throughput anomalies;
- checkpoint inventory and latest logged step versus latest checkpoint;
- optional external evaluation metrics and sample-level failure examples.

For multi-run comparisons, normalize metric names and compare only values with
the same definition and comparable step selection.

### 4. Write One Canonical Report

Read `references/report-template.md` before drafting the final report.

The report should include:

- analysis scope and decision questions;
- exact artifact paths;
- headline conclusions;
- run metadata and training dynamics;
- checkpoint audit;
- evaluation and comparison tables when evidence exists;
- representative failures when sample-level files exist;
- recommendations and missing evidence.

### 5. Use Multi-Agent Mode Only When Helpful

If the task includes multiple independent workstreams, read
`references/multi-agent-plan.md`. Use subagents only when explicitly available
and useful for parallel artifact discovery, metric extraction, comparison, or
error analysis.

## Output

- State the report path if a file was written.
- State which run directories and external artifacts were analyzed.
- Summarize the strongest conclusions and biggest evidence gaps.
- State which extraction commands ran and whether any artifacts were missing.

## Read On Demand

- `references/mvp-engine-artifacts.md`: artifact layout and discovery commands.
- `references/report-template.md`: final Markdown report structure.
- `references/multi-agent-plan.md`: optional parallel analysis split.
