
# Qwen2.5-VL

`qwen2_5_vl` is a minimal image-text training recipe for
`Qwen/Qwen2.5-VL-7B-Instruct`. Its training flow follows the same engine and kit
boundaries used by `recipes/qwen3_vl`, while model-specific packed positions and
vision FLOPs are implemented locally for Qwen2.5-VL.

This first version intentionally supports image data only. It uses the shared
`MLLMDataKit` default `MLLMSampleKit` and `MLLMMediaKit`; video fields are
rejected with an explicit error so timestamp metadata cannot be mixed into image
training.

## Data

Download the development dataset:

```bash
hf download mvp-lab/mvp-engine-vlm-dev-data \
  --repo-type dataset \
  --local-dir data/mvp-engine-vlm-dev-data
```

The default stage configs read:

```text
data/mvp-engine-vlm-dev-data/meta.json
```

Rows should provide multimodal chat data with image references under `images`
and image sizes under `image_size` or `img_size`.

## Stages

- `configs/stage1.yaml`: projector alignment, with ViT and LLM frozen.
- `configs/stage2.yaml`: full pretraining.
- `configs/stage3.yaml`: full SFT with the loss spike guard enabled.

## Run

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/qwen2_5_vl/configs/stage1.yaml
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/qwen2_5_vl/configs/stage2.yaml
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/qwen2_5_vl/configs/stage3.yaml
```

For one-GPU smoke validation on the cluster:

```bash
srun -p gpu -A proj_agent --gres gpu:h200:1 \
  pytest recipes/qwen2_5_vl/tests/test_smoke.py -q --run-smoke --world-size=1
```

