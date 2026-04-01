# TomatoViT Shared-Contract Drift Example

Use this example when the target recipe carries substantial custom engine, dataset, and checkpoint logic of its own, and the upstream shared layer has evolved in config, parallelism, or runtime contracts.

## Why this is a strong merge-repair example

`recipes/tomatovit/` is not a lightweight recipe that only swaps model settings. It customizes:

- its own engine lifecycle and training logic
- a WebDataset + DALI data path
- extra state such as a teacher model, PartialFC heads, and iBOT loss
- custom save/load/checkpoint behavior
- multiple stage configs and parallel configs

That makes it a realistic example of a common merge-repair pattern: failures do not land in one config key only. They show up across entrypoints, runtime paths, auxiliary state, and restore paths at the same time.

## Drift signals you can already see in the current repository

1. `recipes/tomatovit/engine/tomatovit_engine.py` does not define a recipe-local `ConfigClass`, but the engine reads many recipe-specific fields such as `data.*`, `model.*`, `optim.compile_*`, and `model.load_from.*`.
2. There is no checked-in `schema.py` under `recipes/tomatovit/configs/`, while `mvp_engine/engine/engine.py` validates configs through `ConfigClass` before the engine runs. If the recipe keeps inheriting `BaseEngineConfig`, those recipe-specific fields are dropped before they reach the engine.
3. The engine still depends on `self.parallel_backend`, but the current shared `Engine` does not expose that attribute.
4. The same engine still calls `parallelize_model(..., backend=...)`, while the current `mvp_engine/distributed/parallelize.py` only accepts `model`, `device_mesh`, and `backend_kwargs`.
5. The recipe’s custom checkpoint flow still keeps core settings under `loop.checkpoint` and still uses the older `save_checkpoint(...)` / `load_checkpoint(...)` calling pattern, while the shared engine and shared checkpoint helpers are organized around top-level `checkpoint` config and mesh-driven semantics.
6. `stage1_fsdp.yaml`, `stage1_tp.yaml`, and `stage1_fsdp2_tp.yaml` still use older mesh keys such as `dp_size`, `fsdp2_size`, and `tp_size`, and still use the old `backend_kwargs` layout.

These signals tell you this is not a “fix one import and move on” situation. The target recipe has multiple outdated assumptions about the shared layer at once.

## How the skill should use this example

For this kind of recipe, the skill should not jump straight into textual conflict resolution. It should first build a hotspot map:

- which issues belong to config entrypoints and schema drift
- which issues come from the engine depending on old shared interfaces
- which issues belong to runtime gaps such as parallelism, checkpointing, and restore flows
- which issues come from recipe-owned extra state such as teacher/head/scheduler/loss handling

In other words, this example should remind the agent that a complex recipe often needs “entrypoint + runtime + restore-path” repair, not just single-point patching.

## A good repair order to extract from this example

1. Restore the config entrypoint first: confirm that recipe-local schema, engine `ConfigClass`, and YAML layout still deliver the right fields into the engine.
2. Then repair contract drift between the engine and shared helpers: parallelism, checkpointing, runtime entrypoints, compile/optimizer hooks, and related call sites.
3. Then repair the recipe-owned auxiliary state paths: teacher model, auxiliary heads, loss modules, schedulers, and custom load/save behavior.
4. Finish with targeted post-merge validation instead of treating merge completion as proof.

## The most important lesson from this example

- Complex recipe breakages often span multiple layers and cannot be repaired one error at a time.
- If a recipe wrapped or reimplemented shared capabilities locally, those “second-layer wiring” points are where shared-layer evolution will usually break first.
- Merge repair is not only about making the main model run again; it must also cover auxiliary state, restore paths, and long-running training entrypoints.
