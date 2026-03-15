# Model

**模型接入与转换** 类 skill（如新模型接入仓库、权重格式转换、模型注册）。步骤类似但会涉及模型特定结构。

## Skill 列表

- **model-migration**：将外部模型迁移到 `mvp-engine/recipes/<recipe>/model/`，并执行输出一致性与 checkpoint 严格兼容性校验。
  - 路径：[model/model-migration/SKILL.md](model-migration/SKILL.md)
- **model-flops-utilization**：为 Transformer/ViT 模型实现并校验 `calculate_model_flops(...)`，用于 MFU 统计（`MFU = model_flops / device_peak_flops`），并显式记录计算假设。
  - 路径：[model/model-flops-utilization/SKILL.md](model-flops-utilization/SKILL.md)
