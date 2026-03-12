# Training

**训练技巧** 类 skill（需按模型适配，如 gradient checkpointing、自定义 loss 接入）。模式固定但实现依赖各模型结构。

- `model-compile`：为 recipe 接入或调整 `model.compile`，包括 compile 顺序选择、额外模块覆盖和最小验证流程。
