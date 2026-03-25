# ViT Classification Template

This recipe focuses on a plain ImageNet-style classification example with
`google/vit-base-patch16-224`.

## What it includes

- `dataset/imagenet.py`: standard ImageNet transforms plus an optional `FakeData`
  path for smoke runs.
- `model/vit.py`: builds the HuggingFace ViT classifier. The template defaults to
  local random initialization so users do not need to download weights up front.
- `engine/vit_classification_engine.py`: minimal train/evaluate loop on top of the
  shared `Engine`.
- `configs/train.yaml`: a small, readable starter config.

## Expected real dataset layout

Use the standard `ImageFolder` layout when switching `data.use_fake_data=false`:

```text
data/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

## Run

```bash
srun -p camera-long --gres gpu:h200:1 \
.venv/bin/torchrun --nproc_per_node=1 -m mvp_engine.launch \
  --config ./recipes/vc_debug/configs/train.yaml \
  loop.total_steps=2 data.batch_size=8 data.num_workers=0 \
  parallel.mesh.replicate=1 parallel.mesh.shard=1 parallel.mesh.tensor=1 \
  log.interval=1
```

For a real pretrained initialization, set `model.load_pretrained_weights=true`.
With the default H200 `bf16` config, the training log will include `mfu=...`.
