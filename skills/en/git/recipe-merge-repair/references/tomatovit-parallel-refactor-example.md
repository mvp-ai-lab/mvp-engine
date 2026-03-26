# Tomatovit Parallel Refactor Example

Use this example when a recipe broke after shared config and distributed refactors.

## Incoming changes to inspect

- `c3e6ef1 enhance (refactor): a better config system (#56)`
- `c37c2a7 support fsdp2 params cpu offload (#60)`
- `196b919 fix ckpt creation bug (#57)`

These commits changed repo-wide contracts that `recipes/tomatovit/` depended on.

## Breakages found in `recipes/tomatovit/`

1. `TomatoViTEngine` still relied on `BaseEngineConfig`, so recipe-only fields such as `data.*`, `model.*`, and `optim.compile*` were dropped during validation.
2. The engine still called `parallelize_model(..., backend=...)`, but the shared helper no longer accepts a backend argument.
3. The engine still expected `self.parallel_backend` and old checkpoint helper signatures.
4. `stage1_fsdp.yaml`, `stage1_tp.yaml`, and `stage1_fsdp2_tp.yaml` still used old mesh keys: `dp_size`, `fsdp2_size`, `tp_size`.
5. The old flat `parallel.backend_kwargs` layout no longer matched the repo schema.
6. `stage1.yaml` and `stage2.yaml` still stored checkpoint settings under `loop.checkpoint` instead of the top-level `checkpoint` block.
7. TP-enabled configs could not work because `TomatoViTModel` defined no `TP_MODULE_CONFIG`.

## Repair shape

- Add `recipes/tomatovit/configs/schema.py` with recipe-local Pydantic models.
- Set `ConfigClass` on `TomatoViTEngine`.
- Update the engine to:
  - route DDP vs FSDP2 from `DeviceMesh`
  - call `parallelize_model(...)` with the current signature
  - call checkpoint helpers with `mesh` first
- Migrate the YAML files to current mesh/backend layout.
- Add recipe-local TP module config for:
  - `TomatoViTFlashAttention2`
  - `TomatoViTMoTFlashAttention2`
  - `SiglipMLP`

## Validation shape

- Compile the recipe Python tree.
- Run a config/schema smoke test that proves the recipe fields survive validation.
- Run a GPU smoke test that:
  - allocates a GPU with the local cluster command or alias
  - activates `.venv`
  - builds a temporary local pretrained fixture
  - instantiates the recipe model/engine on the updated config path

## Why this belongs in a skill

- The workflow is stable: inspect merged contracts, map recipe dependencies, repair locally, validate.
- The actual fix is recipe-specific: schema fields, mesh layout, TP plan, and engine wiring depend on the recipe.
