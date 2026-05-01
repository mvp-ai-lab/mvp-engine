---
name: experiment-analysis
description: Analyze mvp-engine experiment outputs and turn them into a structured Markdown report. Use when the user wants a benchmark summary, run comparison, regression diagnosis, checkpoint audit, or experiment writeup grounded in outputs/<run_id>/ artifacts.
---

# Experiment Analysis

## Goal

- Produce one canonical Markdown report instead of a loose summary.
- Ground every conclusion in concrete artifact paths.
- Use repo-native run artifacts as primary inputs and external evaluation files as optional evidence.
- Make the report directly useful for experiment decisions, not just archival.

## Required Inputs

- One or more run directories under `outputs/`.
- The standard run artifacts when they exist:
  - `config_<run_id>.yaml`
  - `log_<run_id>.log`
  - `checkpoints/`
- Optional external evaluation artifacts such as:
  - `results*.json`
  - `metrics*.json`
  - `samples*.jsonl`
  - `predictions*.jsonl`
  - `predictions*.csv`
- Optional baseline or reference runs for comparison.
- An existing target report file when the user wants one canonical document updated in place.

## Workflow

### 1. Frame the report as an experiment decision document

- By default, the report should try to answer these questions:
  - how strong is the overall performance
  - which categories, tasks, durations, or buckets are strong or weak
  - what representative failures look like
  - what should change next
- If the available artifacts cannot support one of those questions, say so explicitly and name the missing files.

### 2. Start from the repo's actual run artifacts

- Treat these as the stable primary input set:
  - `outputs/<run_id>/config_<run_id>.yaml`
  - `outputs/<run_id>/log_<run_id>.log`
  - `outputs/<run_id>/checkpoints/`
- Treat downstream evaluation files as optional secondary evidence.

### 3. Collect inputs in a fixed order

- Gather evidence in this order:
  1. the target run directory under `outputs/`
  2. `config_<run_id>.yaml`
  3. `log_<run_id>.log`
  4. `checkpoints/`
  5. external evaluation artifacts
  6. optional baseline or reference runs
- If later-stage artifacts are missing, continue with the strongest evidence available and record the gap.

### 4. Prefer deterministic extraction before writing

- Run the bundled helper script on each run directory before drafting the report:

```bash
python3 skills/experiment/analysis/scripts/summarize_run.py \
  --run-dir outputs/<run_id>
```

- Use the script output as report notes, not as a substitute for reading the raw log when anomalies or message context matter.

### 5. Build the report from evidence

- Record the exact paths for config, logs, checkpoints, and any external evaluation files.
- Summarize run setup, including recipe, engine, workflow, model name, loop policy or total steps, and git info when present.
- Summarize training dynamics from the log:
  - latest metrics
  - best metrics
  - instability, stalls, or regressions
  - throughput or ETA patterns when relevant
- Audit checkpoint behavior:
  - which checkpoints exist
  - whether the last logged step has a matching checkpoint
  - whether retention likely removed earlier checkpoints
- When external evaluation artifacts exist, add:
  - overall scores
  - category or bucket breakdowns
  - comparison tables versus baselines
  - representative error examples and repeated failure modes
  - recommendations grounded in those findings

### 6. Apply a strict evidence standard

- Distinguish direct evidence from inference.
- Do not invent benchmark-level scores from training logs.
- If only training logs exist, frame the report as run analysis rather than benchmark evaluation.
- If requested analyses are blocked by missing sample-level or grouped metrics, state that explicitly.

### 7. Handle multi-run comparison carefully

- Summarize each run separately before building comparison tables.
- Normalize metric names and keep model or run labels consistent.
- Compare only values derived from the same metric definition and comparable step selection.

### 8. Maintain one canonical report

- Prefer updating one report file instead of creating duplicates.
- If the user gives a target document, extend that file.
- Keep tables reproducible from artifact paths.
- When both repo-native and external evaluation artifacts exist, say which conclusions come from which source.

## Validation

- Every conclusion is grounded in exact artifact paths.
- Direct evidence and inference are labeled distinctly.
- Missing files or blocked analyses are called out explicitly.
- The report does not invent benchmark scores from training-only logs.
- Multi-run tables use normalized metric names and comparable measurements.

## Output

- Deliver one Markdown report that includes:
  - analysis scope
  - purpose and decision questions
  - input artifact paths
  - overall performance conclusions
  - category, task, duration, or bucket findings when available
  - run metadata and training dynamics
  - checkpoint inventory or audit notes
  - comparison deltas when comparison targets exist
  - representative error examples and failure analysis when sample-level data exists
  - concrete next-step recommendations

## Read On Demand

- Read [references/mvp-engine-artifacts.md](references/mvp-engine-artifacts.md) when you need the exact artifact patterns, search commands, or output layout.
- Read [references/report-template.md](references/report-template.md) before drafting the final Markdown report.
- Read [references/multi-agent-plan.md](references/multi-agent-plan.md) only when multi-agent mode is available and the task benefits from splitting the analysis.
