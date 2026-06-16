# Qwen3-VL

`qwen3_vl` is a Qwen3-VL-8B-Instruct full-SFT recipe for MVP Engine. Qwen3-VL is
Qwen's vision-language model family for image-text-to-text chat, visual
perception, OCR, spatial reasoning, long-context multimodal understanding, and
agent-style interaction.

This recipe keeps the training entrypoint intentionally direct:

- model: `Qwen/Qwen3-VL-8B-Instruct`
- data: `mvp-lab/mvp-engine-vlm-dev-data`
- config: `configs/train.yaml`
- training mode: full SFT, with ViT, projector, and LLM all trainable
- data path: Lance data through `mvp_dataset` and the standard `MLLMDataKit`
- runtime features: packed attention preparation, token-normalized loss, MFU
  logging, FSDP2-friendly defaults, and shared checkpointing

Model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

## Data

Download the development dataset before running:

```bash
hf download mvp-lab/mvp-engine-vlm-dev-data \
  --repo-type dataset \
  --local-dir data/mvp-engine-vlm-dev-data
```

The default config reads:

```text
data/mvp-engine-vlm-dev-data/meta.json
```

The dataset should provide multimodal chat rows with image references under the
`images` ref column.

## Default Config

The default config is intended for 8 H200-class GPUs HSDP:

- `parallel.mesh.replicate=-1`
- `parallel.mesh.shard=8`
- `parallel.mesh.tensor=1`
- `optim.global_batch_size=256`
- `loop.total_steps=-1`, inferred from the packed dataset
- `checkpoint.hf_enable=true`

## Run

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/qwen3_vl/configs/train.yaml
```

Common overrides:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/qwen3_vl/configs/train.yaml \
  data.train_path=./data/mvp-engine-vlm-dev-data/meta.json \
  model.pretrained_model_name_or_path=Qwen/Qwen3-VL-8B-Instruct
```
