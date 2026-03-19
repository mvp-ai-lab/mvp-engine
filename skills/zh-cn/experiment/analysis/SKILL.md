---
name: experiment-analysis
description: 分析 `mvp-engine` 的实验产物并整理成结构化报告。适用于用户希望基于 `outputs/<run_id>/config_*.yaml`、`log_*.log`、`checkpoints/` 以及可选的外部评测文件（如 `results.json`、`samples*.jsonl`、预测文件）生成 benchmark 总结、run 对比、回归诊断或实验结论文档。
---

# Experiment Analysis

把实验产物整理成报告，而不是松散摘要。  
**English:** [SKILL.md](../../../en/experiment/analysis/SKILL.md)

## 目标

- 产出一个唯一、可持续更新的 Markdown 报告。
- 所有结论都落到明确的文件路径上。
- 以 run 目录下的标准产物为主输入，以外部评测文件为可选补充输入。
- 让报告直接服务于实验判断，而不是只做结果归档。

## 这份报告要回答什么

默认把这份报告当成一次实验复盘和后续决策文档。它至少要尝试回答 4 个问题：

1. 综合性能怎么样。
   - 当前 run / 模型整体是否比 baseline 更好。
   - 是否值得保留、继续训练或继续做下游分析。
2. 每一类表现怎么样。
   - 哪些 task / category / duration / bucket 是强项，哪些是短板。
   - 综合分数背后，具体是哪些维度在拉高或拉低结果。
3. 错在什么地方。
   - 要给出代表性错误示例，而不只是说“某类比较差”。
   - 要总结重复出现的失败模式和可能成因。
4. 下一步该怎么改。
   - 给出可执行的改进建议，如补什么数据、补什么导出、改什么训练设置、优先看哪些 bucket。

如果输入不够支持其中某一项，就在报告里明确写出“当前无法回答什么”以及缺失了哪些文件。

## 1. 先从本仓库真实产物出发

这个仓库目前稳定可依赖的 run 输出通常是：

- `outputs/<run_id>/config_<run_id>.yaml`
- `outputs/<run_id>/log_<run_id>.log`
- `outputs/<run_id>/checkpoints/`

标准输入以这三类文件为主。`results*.json`、`samples*.jsonl`、`predictions*.jsonl`、`predictions*.csv` 归为可选外部评测产物。

需要看更具体的搜索方式、目录结构和命令时，读 [references/mvp-engine-artifacts.md](references/mvp-engine-artifacts.md)。

## 2. 按这个顺序收集输入

1. `outputs/` 下的目标 run 目录。
2. `config_<run_id>.yaml`：拿运行配置、recipe、engine、git 信息。
3. `log_<run_id>.log`：拿训练指标、warning、error 和运行时信息。
4. `checkpoints/`：确认保存节奏、最新 checkpoint 和保留策略。
5. 可选的外部评测产物，可能来自下游脚本或其他 repo：
   - `results*.json`
   - `metrics*.json`
   - `samples*.jsonl`
   - `predictions*.jsonl`
   - `predictions*.csv`
6. 可选的 baseline / reference run。

后面的输入缺失时，不要停住；明确写出缺失项，然后基于现有证据继续。

## 3. 写报告前先做确定性抽取

写报告前，优先先对每个 run 目录跑一次附带脚本：

```bash
python3 skills/en/experiment/analysis/scripts/summarize_run.py \
  --run-dir outputs/<run_id>
```

这个脚本会抽出：

- 标准产物路径
- checkpoint 清单
- config 摘要
- 日志中的最新 / 最小 / 最大指标
- warning 和 error 计数

把脚本输出当作报告草稿数据，而不是替代原始日志。遇到异常、跳变或可疑结论时，必须回到 `log_<run_id>.log` 复核上下文。

## 4. 报告工作流

1. 记录 config、log、checkpoints 和外部评测文件的准确路径。
2. 先交代 run 基本信息：
   - recipe / engine
   - workflow
   - model name
   - total steps 或 loop policy
   - git info（如果有）
3. 从 `log_<run_id>.log` 总结指标变化：
   - 最新值
   - 历史最好值
   - 是否存在震荡、停滞或回退
   - 有必要时补 throughput / ETA 观察
4. 检查 checkpoint：
   - 当前有哪些 checkpoint
   - 最后一个已记录 step 是否有对应 checkpoint
   - 是否可能因为保留策略删掉了更早的 checkpoint
5. 如果存在外部评测产物，再补：
   - overall 分数
   - category / task / duration / bucket 分数
   - 与 baseline / reference 的对比表
   - 基于 `samples*.jsonl` 或预测文件的代表性错误示例与失败模式
   - 结合上述证据给出改进建议
6. 最终先写成 Markdown。

## 5. 证据标准

- 明确区分直接证据和推断。
- 不要从训练日志臆造 benchmark 分数。
- 如果只有训练日志，就把报告定位成 run analysis，不要假装是 benchmark evaluation。
- 如果用户要 error analysis，但没有样本级输出，就明确写明分析被哪些缺失产物阻塞。
- 如果用户要 category 级性能拆分，但没有分项指标文件，就明确写明当前只能回答 overall 或训练过程问题。

## 6. 报告结构

使用 [references/report-template.md](references/report-template.md)。

默认报告至少覆盖：

- 分析范围
- 报告目的与要回答的问题
- 输入与路径
- 综合性能
- 分类 / 任务 / bucket 维度表现
- run 元信息
- 训练过程与指标变化
- checkpoint 状态
- 若有外部评测，再写对比章节
- 若有样本级文件，再写错误示例与失败模式分析
- 明确、可执行的后续建议

## 7. 多 run 对比

做对比时：

- 先分别总结每个 run
- 建表前统一 metric 命名
- 全文保持一致的模型名和 run 标签
- 确认 diff 来自同一 metric 定义、同一统计口径和同一 step 选择

如果可用 multi-agent，就按 [references/multi-agent-plan.md](references/multi-agent-plan.md) 拆分。

## 8. 文件使用规则

- 优先维护一个主报告，不要平行写多个重复文档。
- 用户指定了目标文件时，直接在那个文件上继续完善。
- 表格必须能追溯到明确的产物路径。
- 一个 run 同时存在仓库原生产物和外部 eval 产物时，要写清楚每个结论来自哪一类输入。

## 9. 输出要求

默认交付物是一个 Markdown 报告，至少包含：

- 精确产物路径
- 基于 config 和 log 的 run 摘要
- 综合性能结论
- 分类 / task / bucket 维度结论
- checkpoint 清单
- 核实过的指标表
- 有对比对象时的差值列
- 有样本级数据时的代表性错误示例与错误分析
- 具体的后续实验建议
