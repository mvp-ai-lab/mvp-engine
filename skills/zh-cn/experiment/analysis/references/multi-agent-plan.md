# Multi-Agent 方案

当任务同时包含下列两项及以上时，优先使用 multi-agent：

- run 产物定位
- 日志指标抽取
- 多 run 对比
- 样本级错误分析
- 最终报告撰写

## 建议拆分

### Agent 1: Run Discovery

负责：

- `outputs/<run_id>/`
- `config_<run_id>.yaml`
- `checkpoints/`

产出：

- 标准路径表
- run 元信息
- checkpoint 清单

### Agent 2: Metric Extraction

负责：

- `log_<run_id>.log`
- 辅助脚本输出

产出：

- 最新指标表
- 最小值 / 最大值指标表
- warning / error 摘要

### Agent 3: Comparison

负责：

- current run 摘要
- baseline / reference 摘要
- 可选的外部 `results*.json`

产出：

- 对齐后的对比表
- diff 列
- 简短解释备注

### Agent 4: Error Analysis

负责：

- `samples*.jsonl`
- `predictions*.jsonl`
- `predictions*.csv`

产出：

- 最难 bucket
- 代表性样例
- 重复失败模式

### Main Agent

负责：

- 最终叙述
- 一致性检查
- 写报告

职责：

- 不要把仓库原生产物和外部 eval 产物混在一起解释
- 模型名和 run 名要前后一致
- 缺失证据要明确标注
- 只维护一个主报告文件
