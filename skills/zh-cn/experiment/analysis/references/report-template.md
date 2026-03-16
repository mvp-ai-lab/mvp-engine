# 报告模板

按需删减章节，但文件路径要尽量靠前写清楚。

## 1. 分析范围与报告目的

- 实验或 benchmark 名称
- 本次分析涉及的 run / model
- 这份报告要回答的具体问题
- 这份报告将用于什么决策

建议显式写出这 4 个问题：

- 综合性能是否达标
- 哪些分类 / 任务 / bucket 最强、最弱
- 代表性错误长什么样
- 下一步最应该改什么

## 2. 输入与路径

| 产物 | 路径 | 用途 |
| --- | --- | --- |
| `config_<run_id>.yaml` | `/abs/path` | run 配置 |
| `log_<run_id>.log` | `/abs/path` | 指标与运行日志 |
| `checkpoints/` | `/abs/path` | checkpoint 清单 |
| `results.json` | `/abs/path` | 可选的评测汇总指标 |
| `samples.jsonl` | `/abs/path` | 可选的样本级分析 |
| `baseline run` | `/abs/path` | 可选对比对象 |

## 3. 结论摘要

- 只保留 3 到 5 条
- 每条都要把“观测到的模式”和“解释”连起来

## 4. 综合性能

优先回答：

- 当前 run 的 overall 表现
- 与 baseline / reference 的总体差异
- 这个综合结果是否支持继续推进

可用表格：

| Model / Run | Overall | Best Step | `vs baseline` | Notes |
| --- | ---: | ---: | ---: | --- |

## 5. 分类 / 任务 / Bucket 表现

这一节不能只放一个总表，要明确指出强项和短板。

可用表格：

| Category / Task / Bucket | Score | `vs baseline` | Rank | Notes |
| --- | ---: | ---: | ---: | --- |

如果只有 task-level，没有 category-level，就如实写清楚。

## 6. Run 元信息

| 字段 | 值 |
| --- | --- |
| Recipe / engine | |
| Workflow | |
| Model name | |
| Loop policy | |
| Total steps | |
| Git info | |

## 7. 训练过程与指标变化

| 指标 | 最新值 | 历史最小值 | 历史最大值 | 备注 |
| --- | ---: | ---: | ---: | --- |

备注里常见可写：

- warmup 是否正常
- 收敛形态
- 是否有震荡或回退
- throughput / ETA 是否异常

## 8. Checkpoint 状态

| Checkpoint | 路径 | 备注 |
| --- | --- | --- |

常见检查点：

- 最后一个日志 step 是否有对应 checkpoint
- 是否缺少预期 checkpoint
- 保留策略是否已经清理掉更早 checkpoint

## 9. 评测对比

只有在存在外部 eval 产物时才写这一节。

| 指标 / 任务 | Baseline | Current | Reference | `Current - Baseline` | `Reference - Current` |
| --- | ---: | ---: | ---: | ---: | ---: |

## 10. 错误示例与失败模式分析

只有在存在样本级输出时才写这一节。

针对每个高优先级 bucket：

- 当前分数或失败数量
- 代表性样例
- 重复出现的失败模式
- 可能原因

建议每个 bucket 至少给 1 到 3 个具体错误示例。

## 11. 建议

- 下一步该补跑或补对比什么
- 现有产物不够时，还需要额外导出什么
- checkpoint 策略是否需要调整
- 哪些数据切片或 benchmark 切片值得重点关注
- 哪些问题优先通过数据改，哪些优先通过训练或模型改

## 12. 备注

- 直接证据
- 推断
- 缺失输入
