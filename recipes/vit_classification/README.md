# ViT Classification Template

This recipe mirrors the `recipes/tomatovit` layout, but keeps the implementation
focused on a plain ImageNet-style classification example with
`google/vit-base-patch16-224`.

## What it includes

- `dataset/imagenet.py`: standard ImageNet transforms plus an optional `FakeData`
  path for smoke runs.
- `model/vit.py`: builds the HuggingFace ViT classifier. The template defaults to
  local random initialization so users do not need to download weights up front.
- `engine/vit_classification_engine.py`: minimal train/evaluate loop on top of the
  shared `Engine`.
- `configs/stage1.yaml`: a small, readable starter config.
- `configs/stage1_tp.yaml`: tensor-parallel example for `tp_size=2`.
- `configs/stage1_fsdp2.yaml`: FSDP2 example for `fsdp2_size=2`.

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
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/vit_classification/configs/stage1.yaml
```

For a real pretrained initialization, set `model.load_pretrained_weights=true`.
