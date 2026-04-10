---
name: experiment-analysis
description: 分析 mvp-engine 的实验产物并整理成结构化 Markdown 报告。适用于基于 outputs/<run_id>/ 下的标准产物生成 benchmark 总结、run 对比、回归诊断、checkpoint 审计或实验结论。
---

# Experiment Analysis

## Goal

- 产出一份唯一的 Markdown 报告，而不是松散摘要。
- 让每个结论都能追溯到明确的产物路径。
- 以仓库原生 run 产物为主输入，以外部评测文件为可选证据。
- 让报告直接服务实验决策，而不只是结果归档。

## Required Inputs

- 一个或多个 `outputs/` 下的 run 目录。
- 这些标准 run 产物（若存在）：
  - `config_<run_id>.yaml`
  - `log_<run_id>.log`
  - `checkpoints/`
- 可选的外部评测文件，例如：
  - `results*.json`
  - `metrics*.json`
  - `samples*.jsonl`
  - `predictions*.jsonl`
  - `predictions*.csv`
- 可选的 baseline 或 reference run。
- 如果用户希望在一个固定文档里持续更新，还需要目标报告文件路径。

## Workflow

### 1. 先把报告定位成实验决策文档

- 默认应尽量回答这些问题：
  - 综合表现到底怎么样
  - 哪些 category、task、duration 或 bucket 强，哪些弱
  - 代表性失败样本是什么样
  - 下一步该改什么
- 如果现有产物不足以支持其中某个问题，就明确写出缺了哪些文件。

### 2. 先从本仓库真实产物出发

- 把这些内容视为稳定的主输入：
  - `outputs/<run_id>/config_<run_id>.yaml`
  - `outputs/<run_id>/log_<run_id>.log`
  - `outputs/<run_id>/checkpoints/`
- 下游评测文件视为可选的补充证据。

### 3. 按固定顺序收集输入

- 按以下顺序取证：
  1. `outputs/` 下的目标 run 目录
  2. `config_<run_id>.yaml`
  3. `log_<run_id>.log`
  4. `checkpoints/`
  5. 外部评测产物
  6. 可选的 baseline 或 reference run
- 如果后续层级的产物缺失，继续基于当前最强证据写报告，并把缺口显式写出来。

### 4. 写报告前先做确定性抽取

- 在起草报告前，对每个 run 目录先跑一次附带脚本：

```bash
python3 skills/en/experiment/analysis/scripts/summarize_run.py \
  --run-dir outputs/<run_id>
```

- 脚本输出可作为报告笔记，但遇到异常、跳变或上下文相关的问题时，仍然要回到原始日志复核。

### 5. 基于证据构建报告

- 先记录 config、log、checkpoint 和外部评测文件的准确路径。
- 再总结 run 基本信息，包括 recipe、engine、workflow、model name、loop policy 或 total steps，以及 git 信息。
- 从日志中总结训练动态：
  - 最新指标
  - 历史最佳指标
  - 震荡、停滞或回退
  - 有必要时的吞吐或 ETA 模式
- 审计 checkpoint：
  - 当前有哪些 checkpoint
  - 最后一个已记录 step 是否有对应 checkpoint
  - 保留策略是否可能删掉了更早的 checkpoint
- 如果存在外部评测产物，再补：
  - overall 分数
  - category 或 bucket 维度拆分
  - 与 baseline 的对比表
  - 代表性错误示例和重复失败模式
  - 基于这些发现的改进建议

### 6. 使用严格的证据标准

- 明确区分直接证据和推断。
- 不要从训练日志臆造 benchmark 分数。
- 如果只有训练日志，就把报告定位成 run analysis，而不是 benchmark evaluation。
- 如果样本级或分组指标缺失导致某类分析无法完成，要明确写出阻塞原因。

### 7. 谨慎处理多 run 对比

- 建对比表前，先分别总结每个 run。
- 统一 metric 命名，并在全文中保持一致的模型名和 run 标签。
- 只比较同一指标定义、同一统计口径和可比 step 选择下的数值。

### 8. 维护单一主报告

- 优先维护一个主报告，而不是并行生成多个重复文档。
- 如果用户指定了目标文档，就直接在那个文件上继续完善。
- 表格必须能追溯到具体产物路径。
- 当同时存在仓库原生产物和外部评测产物时，要写清每个结论来自哪一类输入。

## Validation

- 每个结论都能回溯到明确的产物路径。
- 直接证据与推断有清晰区分。
- 缺失文件或被阻塞的分析都被显式写出。
- 报告不会从训练日志虚构 benchmark 分数。
- 多 run 表格使用了统一后的指标命名和可比数据。

## Output

- 交付一份 Markdown 报告，至少包含：
  - 分析范围
  - 报告目的与决策问题
  - 输入产物路径
  - 综合性能结论
  - 若可用则包含 category、task、duration 或 bucket 结论
  - run 元信息和训练动态
  - checkpoint 清单或审计结论
  - 若有对比对象则包含差值表
  - 若有样本级数据则包含代表性错误示例与失败分析
  - 具体的后续实验建议

## Read On Demand

- 需要精确的产物模式、搜索命令或目录布局时，读取 [references/mvp-engine-artifacts.md](references/mvp-engine-artifacts.md)。
- 在起草最终 Markdown 报告前，读取 [references/report-template.md](references/report-template.md)。
- 只有在支持 multi-agent 且任务确实适合拆分时，才读取 [references/multi-agent-plan.md](references/multi-agent-plan.md)。
