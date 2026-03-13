# Config

Guide for generating recipe config YAML files from user requirements, based on patterns in:
- `recipes/vit_classification/configs/stage1.yaml`
- `recipes/vit_classification/configs/stage1_fsdp2.yaml`
- `recipes/vit_classification/configs/stage1_tp.yaml`

## Goal
Help users create their own `recipe` config YAML with:
- runnable defaults,
- topology-aware parallel mesh settings,
- explicit explanation of why each key is set.

## Required User Inputs (ask when missing)
Before finalizing mesh values, you must ask and confirm these:
1. `num_nodes`
2. `gpus_per_node`
3. GPU memory / model size pressure (rough level: low, medium, high)
4. Whether tensor parallel is desired (`tp_size > 1`) and model divisibility constraints (hidden size / head count)

If any item is missing, do not generate final mesh YAML.

## Hard Gate
Do not output final `parallel.mesh` settings until both are explicitly confirmed by user:
1. TP requirement: whether TP is needed (`tp_size > 1`) or not (`tp_size = 1`)
2. Cluster topology: at least `num_nodes` and `gpus_per_node`

You may provide a draft template, but mark it as `placeholder` and not final.

## Interaction Protocol (mandatory)
If required inputs are missing, send a question message first and stop. Do not continue to final YAML.

Recommended fixed prompt:
```text
To generate the final config, please confirm:
1) TP requirement (needed / not needed)
2) Cluster topology (num_nodes, gpus_per_node)
3) Memory/model pressure (low/medium/high)
```

Recommended user reply format:
```text
TP: needed
num_nodes: 2
gpus_per_node: 8
pressure: medium
```

## Mesh Rule (mesh-driven backend)
Do not output `parallel.type`. Runtime backend is inferred from `parallel.mesh`:
- if `ddp_size` is present, runtime uses DDP, expands it to global `world_size`, and ignores `dp_size` / `fsdp2_size` / `tp_size`
- otherwise runtime uses the FSDP2/TP mesh path
- if `fsdp2_size = -1`, runtime forces `dp_size = 1` and expands `fsdp2_size` to global `world_size`
- if `tp_size = -1`, runtime forces `dp_size = 1` and expands `tp_size` to global `world_size`
- never set both `fsdp2_size = -1` and `tp_size = -1`

For regular FSDP2/TP meshes, always satisfy:
- `world_size = num_nodes * gpus_per_node`
- `dp_size * fsdp2_size * tp_size = world_size`
- Prefer `tp_size * fsdp2_size <= gpus_per_node`
- Prefer `tp_size * fsdp2_size == gpus_per_node` when possible

Recommended strategy:
- Keep **FSDP2 + TP sharding inside each node**
- Use **DP across nodes** (replication between nodes)

That usually means:
- `dp_size ≈ num_nodes`
- `fsdp2_size * tp_size ≈ gpus_per_node`

## Practical Presets
Use these as defaults, then adjust by model constraints:
- Pure DDP: `parallel.mesh.ddp_size: 1` (runtime expands it to `world_size`)
- 1 node x 8 GPUs: `dp=1, fsdp2=4, tp=2` (or `dp=1, fsdp2=8, tp=1` if TP is not needed)
- 2 nodes x 8 GPUs: `dp=2, fsdp2=4, tp=2`
- 4 nodes x 8 GPUs: `dp=4, fsdp2=4, tp=2`
- 1 node x 4 GPUs: `dp=1, fsdp2=2, tp=2`
- 1 node x 2 GPUs: `dp=1, fsdp2=2, tp=1`

If TP divisibility fails (for example hidden size or head count not divisible by `tp_size`), reduce `tp_size` first.

## YAML Generation Workflow
1. Ask required questions first (TP requirement + cluster topology are mandatory).
2. Pick a base from `stage1.yaml`, then override only the sections required by the requested topology.
3. Add/override the `parallel.mesh` section according to confirmed topology, and do not add `parallel.type`.
4. Keep optimizer/loop defaults unless user asks otherwise.
5. Return:
- a complete YAML block
- a short rationale section explaining key choices (`parallel.mesh`, precision, checkpoint interval)

## Output Format for Users
When generating config, always include:
1. `Final YAML`
2. `Explanation` (key -> reason)
3. `Sanity Checks`
- mesh product check
- per-node shard check
- expected DP groups

## Example (2 nodes x 8 GPUs)
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

Why:
- `world_size = 2 * 8 = 16`
- `2 * 4 * 2 = 16`
- `fsdp2 * tp = 8` fits one full node, so sharding stays intra-node
- `dp=2` replicates between the two nodes
