## Training Stages

1. `before_train()` builds dataloaders, model, optimizer, scheduler, scaler, resume state, and timer.
2. `do_train()` iterates batches and calls the step hooks.
3. `after_train()` writes the final checkpoint and closes logging.

Most recipes only need to customize the first two hooks.

### Stage Workflow

The full training path is a small fixed skeleton with recipe hooks inserted at the points where behavior changes.

```text
train()
|-- before_train()
|   |-- prepare_dataloader("train")
|   |-- prepare_model()
|   |-- prepare_optimizer()
|   `-- prepare_scheduler()
|-- do_train()
|   `-- for each batch
|       |-- train_pre_step(ctx)
|       |-- train_exec_step(ctx)
|       |   |-- forward_step(ctx)
|       |   |-- backward_step(ctx)
|       |   `-- optimizer_step(ctx)
|       `-- train_post_step(ctx)
`-- after_train()
    |-- save final checkpoint
    `-- close logger
```
