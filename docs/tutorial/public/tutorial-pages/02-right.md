## Start a ViT Run

Use the included ViT classification recipe for a first training launch:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/vit_classification/configs/train.yaml \
  loop.total_steps=20 \
  data.fake_train_size=128 \
  data.fake_val_size=32
```

This starts the shared launcher, imports the recipe, builds `ViTClassificationEngine`, and trains a ViT-B/16 classifier on synthetic image batches.