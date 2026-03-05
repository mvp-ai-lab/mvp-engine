# TomatoViT Gradient Checkpointing — Full Reference

A **real, verified** example of adding gradient checkpointing to TomatoViT (dual-branch ViT with MoT mixture layers).  
**中文：** [example-tomatovit.zh-CN.md](example-tomatovit.zh-CN.md)

## Model Layout Overview

TomatoViT’s Encoder has three layer kinds:
- **Regular layer** (`TomatoViTEncoderLayer`): single branch, RGB or Depth only.
- **Mixture layer** (`TomatoViTMixtureEncoderLayer`): dual branch, RGB and Depth together.
- **Identity layer** (`TomatoViTIdentityEncoderLayer`): placeholder for index alignment.

This example is useful because it shows how to handle **multiple layer types and multiple input branches**.

## Step 1: Add state in Encoder `__init__`

```python
class TomatoViTEncoder(nn.Module):
    def __init__(self, config: TomatoViTConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        # ... rest of init
```

## Step 2: Checkpointing logic per layer type

### Single-branch layer (RGB or Depth only)

```python
def _forward_single_branch_layer(
    self,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    rotary_pos_emb: Optional[torch.Tensor],
    output_attentions: bool,
    use_gradient_checkpointing: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if use_gradient_checkpointing:
        # custom_forward takes only tensors that need gradients
        # attention_mask and rotary_pos_emb captured in closure
        def custom_forward(states: torch.Tensor) -> torch.Tensor:
            return layer(
                states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                output_attentions=False,  # must be off when checkpointing
            )[0]

        hidden_states = self._gradient_checkpointing_func(custom_forward, hidden_states)
        return hidden_states, None  # no attention weights

    layer_outputs = layer(
        hidden_states,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        output_attentions=output_attentions,
    )
    return layer_outputs[0], layer_outputs[1] if output_attentions else None
```

### Dual-branch mixture layer (RGB + Depth together)

```python
def _forward_mixture_layer(
    self,
    mixture_layer: nn.Module,
    hidden_states: torch.Tensor,
    hidden_states_depth: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    attention_mask_depth: Optional[torch.Tensor],
    rotary_pos_emb: Optional[torch.Tensor],
    rotary_pos_emb_depth: Optional[torch.Tensor],
    output_attentions: bool,
    use_gradient_checkpointing: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if use_gradient_checkpointing:
        # All gradient-carrying inputs as explicit args
        def custom_forward(
            rgb_states: torch.Tensor, depth_states: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return mixture_layer(
                rgb_states,
                depth_states,
                attention_mask,
                attention_mask_depth,
                rotary_pos_emb,
                rotary_pos_emb_depth,
                False,  # output_attentions=False
            )[:2]

        hidden_states, hidden_states_depth = self._gradient_checkpointing_func(
            custom_forward, hidden_states, hidden_states_depth
        )
        return hidden_states, hidden_states_depth, None, None

    layer_outputs = mixture_layer(
        hidden_states, hidden_states_depth,
        attention_mask, attention_mask_depth,
        rotary_pos_emb, rotary_pos_emb_depth,
        output_attentions,
    )
    return layer_outputs
```

## Step 2 (cont): Call from Encoder.forward

```python
def forward(self, hidden_states, hidden_states_depth=None, ...):
    use_gradient_checkpointing = self.gradient_checkpointing and self.training and not output_attentions

    for layer_i, layer in enumerate(self.layers):
        if layer_i in self._mot_layer_to_index:
            # MoT mixture layer
            mixture_layer = self.mixture_layers[self._mot_layer_to_index[layer_i]]
            hidden_states, hidden_states_depth, attn_a, attn_b = self._forward_mixture_layer(
                mixture_layer,
                hidden_states, hidden_states_depth,
                attention_mask, attention_mask_depth,
                rotary_pos_emb, rotary_pos_emb_depth,
                output_attentions,
                use_gradient_checkpointing,
            )
        else:
            # Single-branch RGB layer
            hidden_states, attn = self._forward_single_branch_layer(
                layer, hidden_states,
                attention_mask, rotary_pos_emb,
                output_attentions, use_gradient_checkpointing,
            )
            # Single-branch Depth layer
            if hidden_states_depth is not None:
                depth_layer = self.layers_depth[layer_i]
                hidden_states_depth, attn_depth = self._forward_single_branch_layer(
                    depth_layer, hidden_states_depth,
                    attention_mask_depth, rotary_pos_emb_depth,
                    output_attentions, use_gradient_checkpointing,
                )
    # ...
```

## Step 3: Top-level model (HuggingFace way)

```python
class TomatoViTPreTrainedModel(PreTrainedModel):
    config_class = TomatoViTConfig
    supports_gradient_checkpointing = True  # this is enough
```

`PreTrainedModel.gradient_checkpointing_enable()` will find the Encoder and set its state.

## Enabling in the engine

```python
# recipes/tomatovit/engine/tomatovit_engine.py — inside prepare_model()
gc_enabled = OmegaConf.select(self.config, "model.gradient_checkpointing.enabled", default=False)
gc_use_reentrant = OmegaConf.select(self.config, "model.gradient_checkpointing.use_reentrant", default=False)
if gc_enabled:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc_use_reentrant}
    )
```

## Design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Checkpointing granularity | Once per layer | Balance memory savings vs recompute cost |
| `custom_forward` args | Only gradient-carrying tensors | `checkpoint()` save/recompute only for explicit args |
| `output_attentions` | Force False when checkpointing | Attention weights cannot be recovered on recompute |
| Multiple layer types | One helper each | Keep logic clear; avoid over-abstraction |
| `use_reentrant` | Default False | Compatible with `torch.compile` and non-deterministic ops |
