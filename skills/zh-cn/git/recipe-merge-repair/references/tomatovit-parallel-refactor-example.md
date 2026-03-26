# Tomatovit 并行重构示例

当某个 recipe 在共享 config / distributed 重构之后失效时，优先参考这个例子。

## 先看哪些合入改动

- `c3e6ef1 enhance (refactor): a better config system (#56)`
- `c37c2a7 support fsdp2 params cpu offload (#60)`
- `196b919 fix ckpt creation bug (#57)`

这些 commit 改了全仓共享契约，而 `recipes/tomatovit/` 仍然依赖旧契约。

## 在 `recipes/tomatovit/` 中发现的 breakages

1. `TomatoViTEngine` 还在使用 `BaseEngineConfig`，导致 `data.*`、`model.*`、`optim.compile*` 等 recipe 字段在校验后被静默丢弃。
2. engine 仍然调用 `parallelize_model(..., backend=...)`，但共享 helper 已经不接收 `backend` 参数。
3. engine 仍然依赖 `self.parallel_backend` 以及旧版 checkpoint helper 签名。
4. `stage1_fsdp.yaml`、`stage1_tp.yaml`、`stage1_fsdp2_tp.yaml` 还在使用旧 mesh key：`dp_size`、`fsdp2_size`、`tp_size`。
5. 旧的扁平 `parallel.backend_kwargs` 结构已经不符合当前 schema。
6. `stage1.yaml` 和 `stage2.yaml` 仍然把 checkpoint 放在 `loop.checkpoint` 下，而不是新的顶层 `checkpoint`。
7. 启用 TP 的配置无法工作，因为 `TomatoViTModel` 没有定义 `TP_MODULE_CONFIG`。

## 修复形态

- 新增 `recipes/tomatovit/configs/schema.py`，补 recipe-local Pydantic schema。
- 在 `TomatoViTEngine` 上设置 `ConfigClass`。
- 更新 engine：
  - 从 `DeviceMesh` 推断 DDP/FSDP2
  - 按当前签名调用 `parallelize_model(...)`
  - 用 `mesh` 作为 checkpoint helper 的首参
- 把 YAML 迁移到当前 mesh / backend 布局。
- 给以下模块补 recipe-local TP plan：
  - `TomatoViTFlashAttention2`
  - `TomatoViTMoTFlashAttention2`
  - `SiglipMLP`

## 验证形态

- 先编译整个 recipe Python 目录。
- 跑一个 config/schema smoke test，证明 recipe 字段不会在校验中丢失。
- 再跑 GPU smoke test：
  - 用本地集群命令或 alias 申请 GPU
  - 激活 `.venv`
  - 构造临时本地 pretrained fixture
  - 用更新后的配置路径初始化 recipe model/engine

## 为什么这适合做成 skill

- 工作流是稳定的：先看 merged contract，再映射 recipe 依赖，局部修复，最后验证。
- 具体修法又是 recipe-specific 的：schema 字段、mesh 布局、TP plan、engine 接线都必须按当前 recipe 来写。
