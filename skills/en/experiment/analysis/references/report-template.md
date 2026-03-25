# Report Template

Trim sections that are irrelevant, but keep exact file paths near the top.

## 1. Analysis Scope And Report Purpose

- experiment or benchmark name
- runs or models under analysis
- concrete question the report answers
- what decision this report is meant to support

State these four questions explicitly when possible:

- Is the overall performance strong enough
- Which categories, tasks, or buckets are strongest and weakest
- What representative failures look like
- What should change next

## 2. Inputs And Paths

| Artifact | Path | Purpose |
| --- | --- | --- |
| `config_<run_id>.yaml` | `/abs/path` | run configuration |
| `log_<run_id>.log` | `/abs/path` | logged metrics and messages |
| `checkpoints/` | `/abs/path` | checkpoint inventory |
| `results.json` | `/abs/path` | optional aggregate eval metrics |
| `samples.jsonl` | `/abs/path` | optional sample-level analysis |
| `baseline run` | `/abs/path` | optional comparison target |

## 3. Headline Conclusions

- 3 to 5 bullets only
- each bullet should connect an observed pattern to an interpretation

## 4. Overall Performance

Use this section to answer:

- how strong the current run is overall
- how far it is from baseline or reference
- whether the overall result justifies further investment

Useful table:

| Model / Run | Overall | Best Step | `vs baseline` | Notes |
| --- | ---: | ---: | ---: | --- |

## 5. Category / Task / Bucket Breakdown

This section should make strengths and weaknesses explicit rather than hiding them in one aggregate table.

Useful table:

| Category / Task / Bucket | Score | `vs baseline` | Rank | Notes |
| --- | ---: | ---: | ---: | --- |

If only task-level metrics exist, say that clearly.

## 6. Run Metadata

| Field | Value |
| --- | --- |
| Recipe / engine | |
| Workflow | |
| Model name | |
| Loop policy | |
| Total steps | |
| Git info | |

## 7. Training Dynamics

| Metric | Latest | Best Min | Best Max | Notes |
| --- | ---: | ---: | ---: | --- |

Suggested notes:

- warmup behavior
- convergence shape
- instability or regressions
- throughput or ETA anomalies

## 8. Checkpoint Status

| Checkpoint | Path | Notes |
| --- | --- | --- |

Useful checks:

- latest logged step vs latest checkpoint
- missing expected checkpoints
- retention behavior

## 9. Evaluation Comparison

Only include this section when external eval artifacts exist.

| Metric / Task | Baseline | Current | Reference | `Current - Baseline` | `Reference - Current` |
| --- | ---: | ---: | ---: | ---: | ---: |

## 10. Error Examples And Failure Analysis

Only include this section when sample-level outputs exist.

For each high-priority bucket:

- observed score or failure count
- representative examples
- repeated failure pattern
- likely cause

Aim for 1 to 3 concrete examples per priority bucket.

## 11. Recommendations

- what to rerun or compare next
- what to export additionally if current artifacts are insufficient
- whether the checkpoint strategy should change
- what data or benchmark slices deserve more attention
- which issues are better addressed by data, training, or model changes

## 12. Notes

- direct evidence
- inference
- missing inputs
