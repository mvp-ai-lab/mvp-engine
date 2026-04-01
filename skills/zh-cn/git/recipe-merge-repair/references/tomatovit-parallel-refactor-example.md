# TomatoViT 共享契约漂移示例

当目标 recipe 自己带有较重的 engine / dataset / checkpoint 定制，而上游共享层在配置、并行或运行时契约上发生演进时，优先参考这个例子。

## 为什么它是好的 merge-repair 样本

`recipes/tomatovit/` 不是一个“只换模型配置”的轻量 recipe，它同时定制了：

- 自己的 engine 生命周期与训练逻辑
- WebDataset + DALI 数据路径
- teacher model、PartialFC heads、iBOT loss 等附加状态
- 自己的 save/load/checkpoint 逻辑
- 多份 stage 配置和并行配置

这意味着它很适合拿来演示一种真实场景：merge 失败点不会只落在单个 config 字段上，而是会同时出现在入口、运行路径、辅助状态和恢复链路上。

## 当前仓库里能直接看到的漂移信号

1. `recipes/tomatovit/engine/tomatovit_engine.py` 没有设置 recipe-local `ConfigClass`，但 engine 代码大量访问 `data.*`、`model.*`、`optim.compile_*`、`model.load_from.*` 等 recipe 专属字段。
2. `recipes/tomatovit/configs/` 目录下没有 checked-in 的 `schema.py`，而 `mvp_engine/engine/engine.py` 会先用 `ConfigClass` 做 Pydantic 校验；如果继续沿用 `BaseEngineConfig`，这些 recipe 字段会在进入 engine 前被丢掉。
3. `recipes/tomatovit/engine/tomatovit_engine.py` 仍在依赖 `self.parallel_backend`，但当前共享 `Engine` 并没有提供这个属性。
4. 同一个 engine 仍然调用旧式 `parallelize_model(..., backend=...)`，但当前 `mvp_engine/distributed/parallelize.py` 只接收 `model`、`device_mesh` 和 `backend_kwargs`。
5. 自定义 checkpoint 路径仍然把核心设置放在 `loop.checkpoint` 下，并继续使用旧式 `save_checkpoint(...)` / `load_checkpoint(...)` 调用方式；而共享 `Engine` 和共享 checkpoint helper 已经围绕顶层 `checkpoint` 和 `mesh` 驱动的语义组织。
6. `stage1_fsdp.yaml`、`stage1_tp.yaml`、`stage1_fsdp2_tp.yaml` 仍然使用 `dp_size`、`fsdp2_size`、`tp_size` 这类旧 mesh key，并且 `backend_kwargs` 还是旧布局。

这些信号说明：这不是“修一个 import”就能结束的问题，而是 target recipe 对共享层的多处旧假设已经同时过期。

## 这个例子应该怎样被 skill 使用

面对这种 recipe，skill 的重点不应是直接 merge 冲突块，而应先建立一张 hotspot map：

- 哪些问题属于配置入口与 schema 漂移
- 哪些问题属于 engine 对共享层旧接口的依赖
- 哪些问题属于并行、checkpoint、恢复链路等运行时断层
- 哪些问题来自 recipe 自己额外维护的状态，例如 teacher / head / scheduler / loss

换句话说，这个例子要提醒 agent：复杂 recipe 的 merge repair 往往是“入口 + 运行时 + 恢复链路”的组合修复，而不是单点修补。

## 适合从这里抽取的修复顺序

1. 先让配置入口重新可用：确认 recipe-local schema、engine `ConfigClass` 和 YAML 布局是否还能把字段安全送进 engine。
2. 再处理 engine 与共享层 helper 的契约漂移：并行、checkpoint、运行时入口、compile/optimizer 等调用是否仍然成立。
3. 然后补上 recipe 自己额外维护的状态路径：teacher model、aux heads、loss、scheduler、load/save 逻辑。
4. 最后做针对性的 post-merge 验证，而不是只看 merge 是否结束。

## 这个例子最有价值的经验

- 复杂 recipe 的 breakage 往往跨多个层次，不能只按单个报错逐个修。
- 如果 recipe 自己复制或包了一层共享能力，那么共享层改动之后，最先要检查的就是这些“二次接线”位置。
- merge repair 不只是让主模型跑起来，还要确认附加状态、恢复路径和长期训练入口没有一起坏掉。
