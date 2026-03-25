# ViT MFU Reference (MFU-only)

This reference keeps only MFU-related code snippets.

## Included pieces

- `model/vit.py`: `inject_model_flops_calculation(...)` and `calculate_model_flops(...)`.
- `engine/vit_classification_engine.py`: `calculate_mfu(...)` and `perf/mfu` logging flow.
- `configs/schema.py`: MFU-related schema (`ViTMFUConfig`, `log.mfu` fields).
- `configs/train.yaml`: MFU-related config excerpt.

## Hidden on purpose

Non-MFU parts (dataset pipeline, full engine loop, optimizer/scheduler, and full config surface) are intentionally hidden to keep this reference focused.
