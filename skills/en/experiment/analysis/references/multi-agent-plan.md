# Multi-Agent Plan

Use multi-agent mode when the task includes more than one of:

- run discovery
- metric extraction from logs
- comparison across multiple runs
- sample-level error analysis
- final report writing

## Suggested Split

### Agent 1: Run Discovery

Own:

- `outputs/<run_id>/`
- `config_<run_id>.yaml`
- `checkpoints/`

Produce:

- canonical path table
- run metadata
- checkpoint inventory

### Agent 2: Metric Extraction

Own:

- `log_<run_id>.log`
- helper script output

Produce:

- latest metric table
- min and max metric table
- warning and error summary

### Agent 3: Comparison

Own:

- current run summary
- baseline or reference summaries
- optional external `results*.json`

Produce:

- aligned comparison tables
- delta columns
- brief interpretation notes

### Agent 4: Error Analysis

Own:

- `samples*.jsonl`
- `predictions*.jsonl`
- `predictions*.csv`

Produce:

- hardest buckets
- representative examples
- repeated failure modes

### Main Agent

Own:

- final narrative
- consistency checks
- writing the report

Responsibilities:

- make sure repo-native outputs and external eval outputs are not conflated
- keep model and run naming consistent
- mark missing evidence explicitly
- save one canonical report
