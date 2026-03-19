# minimal-vlm

This recipe is a minimal starting point for training a vision-language model with `mvp-engine`.

## Task Summary

A minimal recipe that shows how to use `mvp-engine` to train a vision-language model (VLM).

The scaffold keeps the engine, dataset, and model code explicit so you can see where image inputs, text inputs, loss computation, and optimizer setup belong.

## What you still need to fill in

- `dataset/`: dataset construction and multimodal batch collation
- `model/`: VLM definition or wrapper around an existing model
- `engine/minimal_vlm_engine.py`: image/text preprocessing, forward pass, loss, and logging
- `configs/train.yaml`: recipe-specific overrides for the minimal VLM setup

## Run

After filling in the stubs, launch with:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/minimal_vlm/configs/train.yaml
```

Use existing recipes as references once you are ready to implement the concrete VLM pieces.
