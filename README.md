<p align="center">
  <picture>
    <img alt="MVP Engine" src="./assets/logo.png" width="600" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
  <strong>Fully Open and Easy-to-Use Framework for Democratized Multimodal Model Training</strong>
  <p align="center">
    by MVP Lab.
  </p>
</p>

## Overview

MVP Engine is a lightweight, extensible training engine for multimodal model research. It provides a clean
pipeline for distributed training (DDP), mixed precision, gradient accumulation, checkpointing, and pluggable
logging backends. The core design focuses on separating **experiment logic** (your model, optimizer, scheduler,
data pipeline) from **training orchestration** (loop policy, logging, checkpointing), so you can iterate on
multimodal ideas without rewriting boilerplate.

## Design at a Glance

- **Engine as the orchestration layer**: `mvp_engine/engine/engine.py` defines the base `Engine` class and the
  train workflow (`before_train -> run_train -> after_train`). Subclasses implement `prepare_*` methods and
  the evaluation pipeline.
- **Registry-based extensibility**: `ENGINE_REGISTRY` makes it easy to register custom engines and select them
  in config via `engine: YourEngine`.
- **Hydra configuration**: `mvp_engine/launch.py` merges default config with recipe configs and launches the
  requested workflow (`train`, `evaluate`, or custom).
- **Logging system**: metrics are aggregated and dispatched to terminal/file backends; additional backends can
  be added with minimal changes.
- **WebDataset data pipeline**: `mvp_engine/dataset/webdataset.py` provides a resampled shard loader for large
  scale multimodal datasets stored in tar shards.

### Training Workflow

1. **Initialize**: setup parallel, seed, run ID, output directory, and loggers.
2. **Build components**: dataloaders, model, optimizer, scheduler, gradient scaler.
3. **Run loop**: iteration-based training with optional gradient accumulation and mixed precision.
4. **Checkpoint**: periodic and final checkpoints, plus engine state for resuming.

### Project Layout

- `mvp_engine/engine/` — core orchestration logic and Engine base class
- `mvp_engine/utils/` — logging, distributed helpers, training utilities
- `mvp_engine/dataset/` — dataset builders (WebDataset utilities)
- `recipes/` — experiment configs and custom engine/model definitions
- `outputs/` — run outputs, logs, and checkpoints


## Getting Started

```
mkdir data
cd data
ln -s /mnt/data-alpha-sg-02/team-camera/projects/Potato3D/processed_data/potato_v1 ./

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355 -m mvp_engine.launch --config ./recipes/tomatovit/configs/stage1.yaml
```

## Development

```
uv venv --python=3.12
source .venv/bin/activate

# For Dependencies
uv sync
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# For Development Tools
uv pip install pre-commit
pre-commit install
```
