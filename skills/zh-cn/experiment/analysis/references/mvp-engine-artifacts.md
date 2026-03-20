# `mvp-engine` 产物说明

这个 skill 的输入分成两类：仓库原生产物，以及外部评测产物。

## 仓库原生的标准产物

一次正常运行里，最可靠的文件通常在：

```text
outputs/<run_id>/
├── config_<run_id>.yaml
├── log_<run_id>.log
└── checkpoints/
    ├── iter_<step>/
    └── ...
```

这些文件主要来自：

- `mvp_engine/engine/engine.py`
- `mvp_engine/utils/log/backend/file.py`

## 输入分类

标准输入：

- `config_<run_id>.yaml`
- `log_<run_id>.log`
- `checkpoints/`

可选外部评测输入：

- `results*.json`
- `metrics*.json`
- `samples*.jsonl`
- `predictions*.jsonl`
- `predictions*.csv`

## 常用搜索命令

找仓库原生产物：

```bash
rg --files outputs -g 'config_*.yaml' -g 'log_*.log'
find outputs -path '*/checkpoints/*' -maxdepth 3 -type d
```

找 run 附近的外部评测产物：

```bash
rg --files outputs -g 'results*.json' -g 'metrics*.json' -g 'samples*.jsonl' -g 'predictions*.jsonl' -g 'predictions*.csv'
```

对单个 run 做确定性摘要：

```bash
python3 skills/en/experiment/analysis/scripts/summarize_run.py \
  --run-dir outputs/<run_id> \
  --output outputs/<run_id>/run_summary.json
```

## 从日志能稳定得到什么

`log_<run_id>.log` 一般可以支持：

- 最新日志指标
- 历史最小值 / 最大值
- step 覆盖范围
- warning / error 数量
- 显式写入日志的运行信息

但它本身不能替代分项评测文件或样本级错误文件。

## 报告规则

如果只有仓库原生产物，就写 **run analysis** 报告。

如果同时有外部 eval 产物，再在此基础上扩展成 **evaluation** 报告。
