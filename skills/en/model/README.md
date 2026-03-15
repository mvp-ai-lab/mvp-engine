# Model

Skills for **model integration and conversion** (e.g. adding a new model to the repo, weight conversion between formats, model registration). The steps are similar across models but the code touches model-specific structure.

## Skill List

- **model-migration**: Migrate an external model into `mvp-engine/recipes/<recipe>/model/` with parity checks and strict checkpoint compatibility.
  - Path: [model/model-migration/SKILL.md](model-migration/SKILL.md)
- **model-flops-utilization**: Implement and validate `calculate_model_flops(...)` for Transformer/ViT models to support MFU reporting (`MFU = model_flops / device_peak_flops`) with explicit assumptions.
  - Path: [model/model-flops-utilization/SKILL.md](model-flops-utilization/SKILL.md)
