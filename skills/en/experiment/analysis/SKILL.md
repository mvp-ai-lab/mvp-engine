---
name: experiment-analysis
description: Analyze `mvp-engine` experiment outputs and turn them into a structured report. Use when the user wants a benchmark summary, run comparison, regression diagnosis, checkpoint audit, or experiment writeup from `outputs/<run_id>/config_*.yaml`, `log_*.log`, `checkpoints/`, and optional external eval artifacts such as `results.json`, `samples*.jsonl`, or prediction files.
---

# Experiment Analysis

Turn experiment artifacts into a report, not a loose summary.  
**中文：** [SKILL.md](../../../zh-cn/experiment/analysis/SKILL.md)

## Goal

- Produce a single canonical Markdown report.
- Ground every conclusion in exact artifact paths.
- Use the run directory's standard artifacts as primary inputs and external eval files as optional inputs.
- Make the report directly useful for experiment decisions, not just archival.

## What this report is for

Treat the report as an experiment review and decision document. By default it should try to answer four questions:

1. How strong is the overall performance.
   - Is the current run or model good enough overall.
   - Is it worth keeping, extending, or comparing further.
2. How each category performs.
   - Which tasks, categories, durations, or buckets are strong.
   - Which dimensions are dragging the overall result down.
3. What the concrete failures look like.
   - Include representative error examples, not only score tables.
   - Summarize repeated failure modes and likely causes.
4. What to change next.
   - Give actionable next steps such as data changes, export gaps, training changes, or priority buckets.

If the available artifacts cannot support one of these questions, state that explicitly in the report and name the missing files.

## 1. Start from the repo's actual artifacts

In this repo, the stable run outputs are usually:

- `outputs/<run_id>/config_<run_id>.yaml`
- `outputs/<run_id>/log_<run_id>.log`
- `outputs/<run_id>/checkpoints/`

Treat these three artifact types as the standard input set. Treat `results*.json`, `samples*.jsonl`, `predictions*.jsonl`, and `predictions*.csv` as optional external eval inputs.

Read [references/mvp-engine-artifacts.md](references/mvp-engine-artifacts.md) when you need the exact file patterns, search commands, or output layout.

## 2. Collect inputs in this order

1. The run directory under `outputs/`.
2. `config_<run_id>.yaml` for run settings, recipe, engine, and git info.
3. `log_<run_id>.log` for training metrics, warnings, errors, and any logged runtime details.
4. `checkpoints/` to verify save cadence, latest checkpoint, and retention behavior.
5. Optional external eval artifacts saved by downstream scripts or another repo:
   - `results*.json`
   - `metrics*.json`
   - `samples*.jsonl`
   - `predictions*.jsonl`
   - `predictions*.csv`
6. Optional baseline or reference runs for comparison.

If later-stage artifacts are missing, say so explicitly and continue with the strongest evidence available.

## 3. Prefer deterministic extraction before writing

Before drafting the report, run the bundled helper script on each run directory:

```bash
python3 skills/en/experiment/analysis/scripts/summarize_run.py \
  --run-dir outputs/<run_id>
```

The script extracts:

- canonical artifact paths
- checkpoint inventory
- config summary
- latest, min, and max logged metrics
- warning and error counts

Use the JSON output as report notes, not as a substitute for reading the raw log when the task depends on anomalies or message context.

## 4. Reporting workflow

1. Record the exact paths for config, log, checkpoints, and any external eval files.
2. Summarize the run setup:
   - recipe / engine
   - workflow
   - model name
   - total steps or loop policy
   - git info if present
3. Summarize the logged metric dynamics from `log_<run_id>.log`:
   - latest metric values
   - best observed values
   - notable instability, stalls, or regressions
   - throughput / ETA patterns if they matter
4. Audit checkpoint behavior:
   - which checkpoints exist
   - whether the last logged step has a corresponding checkpoint
   - whether retention likely removed earlier checkpoints
5. If external eval outputs exist, add:
   - overall scores
   - category, task, duration, or bucket-level scores
   - comparison tables vs baseline or reference runs
   - representative error examples and failure analysis from `samples*.jsonl` or prediction files
   - recommendations grounded in those findings
6. Write the report in Markdown first.

## 5. Evidence standard

- Distinguish direct evidence from inference.
- Do not invent benchmark-level scores from training logs.
- If the repo only provides training logs, frame the report as a run analysis, not a benchmark evaluation.
- If error analysis is requested but there are no per-sample outputs, state that the analysis is blocked by missing artifacts.
- If category-level performance is requested but no grouped metrics exist, state that the report can only answer overall or training-dynamics questions for now.

## 6. Report shape

Use [references/report-template.md](references/report-template.md).

The default report should cover:

- analysis scope
- report purpose and decision questions
- inputs and paths
- overall performance
- category, task, or bucket breakdown
- run metadata
- training dynamics
- checkpoint status
- comparison sections when external artifacts exist
- error examples and failure analysis when sample-level artifacts exist
- actionable recommendations

## 7. Multi-run comparison

For comparison tasks:

- summarize each run separately first
- normalize metric names before building tables
- use consistent model and run labels across the whole report
- make sure differences are computed from the same metric definition and step selection

If multi-agent mode is available, use the split in [references/multi-agent-plan.md](references/multi-agent-plan.md).

## 8. File-conscious rules

- Prefer updating one canonical report instead of creating duplicates.
- If the user gives a target document, extend that file.
- Keep report tables reproducible from artifact paths.
- When a run has both repo-native outputs and external eval outputs, state which findings come from which source.

## 9. Output expectations

The default deliverable is a Markdown report with:

- exact artifact paths
- run summary grounded in config and logs
- overall performance conclusions
- category, task, or bucket conclusions
- checkpoint inventory
- verified metric tables
- comparison deltas when comparison targets exist
- representative error examples and failure analysis when sample-level data exists
- concrete next-step recommendations
