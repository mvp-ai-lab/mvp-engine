# OpenBee

`openbee` is the OpenBee recipe for Qwen3-VL 8B training on OpenBee-style
multimodal data. The recipe is implemented locally under `recipes/openbee/`
and uses the shared `mvp_engine` launch, logging, checkpoint, and distributed
training infrastructure.

## Stages

- `configs/stage1.yaml`: alignment stage Hydra entrypoint.
- `configs/stage2.yaml`: pretraining stage Hydra entrypoint.
- `configs/stage3.yaml`: SFT stage Hydra entrypoint.

Stage entrypoints compose shared config groups under `configs/data/`,
`configs/model/`, `configs/model_checkpoint/`, `configs/model_runtime/`,
`configs/optim/`, `configs/parallel/`, `configs/loop/`, and
`configs/checkpoint/`.

Each stage expects a Lance dataset `meta.json` from `mvp_dataset` and a local
checkpoint path. Override `data.train_path` and
`model.pretrained_model_name_or_path` when using different data or checkpoints.

## Data

The dataset rows should provide:

- `messages` or `conversations`: user/assistant chat turns.
- `images`: image references consumed by `<image>` placeholders.
- `image_size` or `img_size`: image size metadata matching `images`.

The loader validates the raw rows, tokenizes conversations with the Qwen3-VL
processor, always packs samples, resolves image references, and materializes
pixel tensors for training.

## Run

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/openbee/configs/stage1.yaml
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/openbee/configs/stage2.yaml
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/openbee/configs/stage3.yaml
```

Example override:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/openbee/configs/stage2.yaml \
  data.train_path=./data/Open-Bee-Lance/stage2/meta.json \
  model.pretrained_model_name_or_path=./pretrained/Qwen3-VL-8B-Base-woDS-stage1
```
