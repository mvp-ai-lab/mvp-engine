# Magic Transformer

This recipe trains the `MagicTransformer` language model from the recipe-local
`model/source_model.py` design on a deterministic fake autoregressive token dataset.

It is intentionally minimal:

- `dataset/` uses recipe-local fake token sequences so the recipe can run
  without external data preparation.
- `model/` exposes the recipe-local `source_model.py` implementation through a
  recipe-local builder so the recipe can use the existing model directly.
- `engine/` wires the model into the shared `mvp_engine` training loop for
  next-token prediction.

Run the recipe with:

```bash
torchrun --nproc_per_node=1 -m mvp_engine.launch --config ./recipes/magic_transformer/configs/train.yaml
```

The default config keeps parallelism conservative and uses fake data so the
recipe is easy to smoke-test locally before adapting it to real data.
