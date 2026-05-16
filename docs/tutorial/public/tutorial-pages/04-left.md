## Build Your Own

A custom engine usually lives in `recipes/<name>/engine/` and is registered with `ENGINE_REGISTRY`.

The minimum surface is:

```python
@ENGINE_REGISTRY.register()
class MyEngine(Engine):
    ConfigClass = MyConfig

    def prepare_dataloader(self, workflow="train"): ...
    def prepare_model(self): ...
    def prepare_optimizer(self): ...
    def prepare_scheduler(self): ...
```

Then add training behavior:

- `train_pre_step(ctx)` do some data preprocess here, for example, moves tensors to `self.device` and normalizes the batch shape.
- `forward_step(ctx)` runs the model forward and writes `ctx.outputs`.
- `ctx.outputs["loss"]` is used for backward.
- `ctx.outputs["logs"]` is sent to the logger.