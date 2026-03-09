# Config

用于根据用户需求生成 recipe 配置 YAML，参考以下样例：
- `recipes/vit_classification/configs/stage1.yaml`
- `recipes/vit_classification/configs/stage1_fsdp2.yaml`
- `recipes/vit_classification/configs/stage1_tp.yaml`

## 目标
帮助用户生成可直接训练的 `recipe` 配置文件，并满足：
- 有可运行的默认值，
- mesh 配置和硬件拓扑匹配，
- 输出时解释每一项“为什么这么设”。

## 必要输入（缺失时必须追问）
在确定 mesh 之前，必须先问并确认以下信息：
1. `num_nodes`（节点数）
2. `gpus_per_node`（每节点 GPU 数）
3. 显存压力 / 模型规模压力（低、中、高）
4. 是否需要 TP（`tp_size > 1`）以及模型可整除约束（如 hidden size / head 数）

如果任一项缺失，不得输出最终 mesh YAML。

## 硬门槛
在用户明确确认以下两项前，不得输出最终 `parallel.mesh`：
1. TP 需求：是否需要 TP（`tp_size > 1`）或不需要（`tp_size = 1`）
2. 集群拓扑：至少给出 `num_nodes` 与 `gpus_per_node`

可以先给占位模板，但必须标注为 `placeholder`，不是最终配置。

## 交互协议（必须执行）
若必需信息缺失，必须先发“提问消息”并停止，不得继续生成最终 YAML。

提问消息建议固定为：
```text
为生成最终配置，请先确认：
1) 是否需要 TP（需要 / 不需要）
2) 集群拓扑（num_nodes, gpus_per_node）
3) 显存/模型压力（低/中/高）
```

用户可按如下格式回复（推荐）：
```text
TP: 需要
num_nodes: 2
gpus_per_node: 8
pressure: 中
```

## Mesh 规则（由 mesh 推导后端）
不要输出 `parallel.type`。运行时后端由 `parallel.mesh` 自动推导：
- 只要出现 `ddp_size`，就进入 DDP，运行时会把它展开为全局 `world_size`，并忽略 `dp_size` / `fsdp2_size` / `tp_size`
- 否则进入 FSDP2/TP 的 mesh 路径
- 若 `fsdp2_size = -1`，运行时会强制 `dp_size = 1`，并把 `fsdp2_size` 展开为全局 `world_size`
- 若 `tp_size = -1`，运行时会强制 `dp_size = 1`，并把 `tp_size` 展开为全局 `world_size`
- 不能同时设置 `fsdp2_size = -1` 和 `tp_size = -1`

对于常规 FSDP2/TP mesh，必须满足：
- `world_size = num_nodes * gpus_per_node`
- `dp_size * fsdp2_size * tp_size = world_size`
- 优先 `tp_size * fsdp2_size <= gpus_per_node`
- 条件允许时优先 `tp_size * fsdp2_size == gpus_per_node`

推荐策略：
- **节点内**做 FSDP2 + TP 分片
- **节点间**用 DP 做副本

通常对应：
- `dp_size ≈ num_nodes`
- `fsdp2_size * tp_size ≈ gpus_per_node`

## 实用默认档位
先用下面默认，再按模型约束调整：
- 纯 DDP：`parallel.mesh.ddp_size: 1`（运行时会自动展开到 `world_size`）
- 1 节点 x 8 卡：`dp=1, fsdp2=4, tp=2`（若不需要 TP，可用 `dp=1, fsdp2=8, tp=1`）
- 2 节点 x 8 卡：`dp=2, fsdp2=4, tp=2`
- 4 节点 x 8 卡：`dp=4, fsdp2=4, tp=2`
- 1 节点 x 4 卡：`dp=1, fsdp2=2, tp=2`
- 1 节点 x 2 卡：`dp=1, fsdp2=2, tp=1`

若 TP 可整除条件不满足（例如 hidden size 或 head 数不能被 `tp_size` 整除），优先降低 `tp_size`。

## YAML 生成流程
1. 先提问并确认必要信息（TP 需求 + 集群拓扑为必选项）。
2. 选择 `stage1.yaml` 作为基础，再按用户要求覆写必要字段。
3. 按确认后的硬件拓扑补齐/覆盖 `parallel.mesh` 段，不要再添加 `parallel.type`。
4. 除非用户明确要求，`optim` 与 `loop` 保持基础默认。
5. 输出时必须包含：
- 完整 YAML
- 配置解释（关键项 -> 原因）

## 输出格式要求
生成配置时固定输出三段：
1. `Final YAML`
2. `Explanation`（每个关键配置对应原因）
3. `Sanity Checks`
- mesh 乘积校验
- 节点内分片校验
- DP 副本组数量预期

## 示例（2 节点 x 8 卡）
```yaml
defaults:
  - stage1

parallel:
  mesh:
    dp_size: 2
    fsdp2_size: 4
    tp_size: 2
  backend_kwargs:
    reshard_after_forward: true
    mp_policy:
      param_dtype: bfloat16
      reduce_dtype: float32
      output_dtype: bfloat16
```

解释：
- `world_size = 2 * 8 = 16`
- `2 * 4 * 2 = 16`
- `fsdp2 * tp = 8`，分片刚好落在单节点内部
- `dp=2`，在两个节点之间做副本
