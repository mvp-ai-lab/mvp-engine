# Gradient Checkpointing Patterns

Use this reference only when the target model's checkpointing path is unclear or
manual adaptation is required.

## Native Support

Many HuggingFace-style models already own the block-level checkpoint logic. In
that case, do not wrap layers yourself. Enable the existing hook before
distributed wrapping:

```python
gc_config = self.config.model.gradient_checkpointing
if gc_config.enabled:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc_config.use_reentrant}
    )
```

Native support is present only when repeated blocks actually route through the
checkpoint function during training. A method named
`gradient_checkpointing_enable` is not enough if nothing in the layer loop reads
the resulting state.

## Manual Support

For plain `nn.Module` models, add the smallest local interface:

```python
import functools
import torch


class Model(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = functools.partial(
            torch.utils.checkpoint.checkpoint,
            **gradient_checkpointing_kwargs,
        )

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
```

In the repeated-block loop:

```python
use_gc = self.gradient_checkpointing and self.training and not use_cache and not output_attentions

for block in self.blocks:
    if use_gc:
        def custom_forward(hidden_states):
            return block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        hidden_states = self._gradient_checkpointing_func(custom_forward, hidden_states)
    else:
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
```

## Rules

- Checkpoint repeated expensive blocks, not cheap heads or preprocessing glue.
- Pass tensors that require gradients as explicit checkpoint inputs.
- Capture masks, ids, cache objects, flags, and static metadata in closures.
- Disable `use_cache` or other KV-cache paths when checkpointing is active.
- Avoid checkpointing paths that return incompatible auxiliary outputs.
- Keep parameter names and output semantics unchanged.
