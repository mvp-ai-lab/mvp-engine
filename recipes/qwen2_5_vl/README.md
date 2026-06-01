
# Qwen2.5-VL

`qwen2_5_vl` is a minimal image-text training recipe for
`Qwen/Qwen2.5-VL-7B-Instruct`.

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

The default config reads:

```text
data/mvp-engine-vlm-dev-data/meta.json
```

Rows should provide multimodal chat data with image references under `images`
and image sizes under `image_size` or `img_size`.

## Default Config

`configs/train.yaml` runs full image-text SFT:

- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- attention: `flash_attention_2`
- trainable modules: ViT, projector, and LLM
- parallelism: 8-way shard mesh by default
- steps: `loop.total_steps=-1`, inferred from the packed dataset

FlashAttention 2 must be installed for the default attention backend. Use
`model.attn_implementation=sdpa` as a runtime override when FlashAttention 2 is
not available.

For projector-only alignment, use overrides such as:

```bash
model.freeze_vit=true model.freeze_llm=true model.freeze_projector=false
```

## Run

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/qwen2_5_vl/configs/train.yaml
```

Short smoke run:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/qwen2_5_vl/configs/train.yaml \
  loop.total_steps=20
```

