# ViT MFU 参考（仅保留 MFU）

这个参考目录只保留与 MFU 直接相关的代码。

## 保留内容

- `model/vit.py`：`inject_model_flops_calculation(...)` 与 `calculate_model_flops(...)`。
- `engine/vit_classification_engine.py`：`calculate_mfu(...)` 以及 `perf/mfu` 日志写入流程。
- `configs/schema.py`：MFU 相关 schema（`ViTMFUConfig`、`log.mfu`）。
- `configs/train.yaml`：MFU 相关配置片段。

## 已隐藏内容

与 MFU 无关的内容（dataset 流程、完整 engine 训练循环、optimizer/scheduler、完整配置）都已刻意隐藏，避免干扰。
