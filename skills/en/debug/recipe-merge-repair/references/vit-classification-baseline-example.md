# ViT Classification Baseline Example

Use this example when you need a known-good baseline for a recipe that is already aligned with the current engine contracts.

## Why this recipe is useful as a baseline

- `recipes/vit_classification/` already has:
  - a recipe-local `ConfigClass`
  - fake-data support for offline smoke runs
  - current `parallelize_model(...)` usage
  - current top-level `checkpoint` config layout

That makes it a good comparison point when another recipe appears broken after merges.

## What the validation found

1. The recipe code paths were already compatible with the current engine/config stack.
2. The checked-in template default was not single-rank friendly:
   - `parallel.mesh.replicate: -1`
   - `parallel.mesh.shard: 8`
   - `parallel.mesh.tensor: 1`
3. On a `WORLD_SIZE=1` smoke run, that layout inferred `replicate=0` and failed during `DeviceMesh` initialization before recipe logic ran.

## Repair shape

- Keep the recipe template runnable by default:
  - change `parallel.mesh.shard` from `8` to `1`
  - keep a short comment explaining that multi-rank FSDP2 users should raise `shard` intentionally
- Make common tuning overrides ergonomic under Hydra struct mode:
  - expose `patch_size`, `num_channels`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, and `num_attention_heads` in `train.yaml`
  - this allows plain overrides like `model.hidden_size=192` during smoke tests

## Validation shape

- `python -m compileall recipes/vit_classification`
- Config/schema composition for `train.yaml`
- GPU smoke with:
  - fake data
  - local model construction (`load_pretrained_weights: false`)
  - single-rank mesh
  - engine startup and one training forward pass

## Practical lesson for the skill

- Not every recipe needs repair code.
- Sometimes the only issue is that the default validation path is invalid for the requested world size.
- The skill should distinguish:
  - shared-contract breakage
  - recipe logic breakage
  - smoke-only mesh/default mismatch
