# Chapter II: Setup

MVP-Engine is meant to keep the core training loop stable while each recipe owns its model, data, and config.

Start from a clean Python 3.12 environment:

```bash
uv venv --python=3.12
source .venv/bin/activate
uv sync
```

After installation, run commands from the repository root. This keeps recipe paths and Hydra config loading simple.

## Check the Pieces

- `mvp_engine/` provides the launcher and shared engine utilities.
- `recipes/vit_classification/` provides a complete ViT example.
- `recipes/vit_classification/configs/train.yaml` controls data, model, optimizer, loop, and parallelism.

The ViT recipe defaults to `FakeData`, so you can smoke-test the training path before preparing ImageNet.
