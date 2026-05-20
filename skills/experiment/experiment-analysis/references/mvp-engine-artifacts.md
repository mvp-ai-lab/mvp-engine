# `mvp-engine` Artifact Map

This skill works with two input classes: repo-native run artifacts and optional external eval artifacts.

## Canonical repo-native outputs

For a normal run, the most reliable files live under:

```text
outputs/<run_id>/
├── config_<run_id>.yaml
├── log_<run_id>.log
└── checkpoints/
    ├── iter_<step>/
    └── ...
```

These come from:

- `mvp_engine/engine/engine.py`
- `mvp_engine/utils/log/backend/file.py`

## Input classes

Standard inputs:

- `config_<run_id>.yaml`
- `log_<run_id>.log`
- `checkpoints/`

Optional external eval inputs:

- `results*.json`
- `metrics*.json`
- `samples*.jsonl`
- `predictions*.jsonl`
- `predictions*.csv`

## Useful discovery commands

Find repo-native artifacts:

```bash
rg --files outputs -g 'config_*.yaml' -g 'log_*.log'
find outputs -path '*/checkpoints/*' -maxdepth 3 -type d
```

Find external eval artifacts near a run:

```bash
rg --files outputs \
  -g 'results*.json' \
  -g 'metrics*.json' \
  -g 'samples*.jsonl' \
  -g 'predictions*.jsonl' \
  -g 'predictions*.csv'
```

Summarize one run deterministically:

```bash
python3 skills/experiment/experiment-analysis/scripts/summarize_run.py \
  --run-dir outputs/<run_id> \
  --output outputs/<run_id>/run_summary.json
```

## What the log file can support

`log_<run_id>.log` can usually support:

- latest logged metrics
- best observed min and max values
- step coverage
- warning and error counts
- runtime notes that were explicitly logged

It cannot replace grouped eval metrics or sample-level error files.

## Reporting rule

If only repo-native artifacts exist, write a **run analysis** report.

If external eval artifacts also exist, extend that into an **evaluation** report.
