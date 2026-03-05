# TomatoViT Gradient Checkpointing 完整参考实现

这是一个**真实且已验证**的例子，展示如何为 TomatoViT（双分支 ViT + MoT 混合层）添加 gradient checkpointing。  
**English:** [example-tomatovit.md](example-tomatovit.md)

## 模型架构概览

TomatoViT 的 Encoder 有三种 layer：
- **Regular layer**（`TomatoViTEncoderLayer`）：单分支，只处理 RGB 或 Depth。
- **Mixture layer**（`TomatoViTMixtureEncoderLayer`）：双分支，同时处理 RGB 和 Depth。
- **Identity layer**（`TomatoViTIdentityEncoderLayer`）：占位，用于对齐索引。

此例特别适合参考**多种 layer 类型和多个输入分支**的写法。

## Step 1: 在 Encoder `__init__` 中添加状态

```python
class TomatoViTEncoder(nn.Module):
    def __init__(self, config: TomatoViTConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        # ... 其余初始化
```

## Step 2: 为每种 layer 类型编写 checkpointing 逻辑

### 单分支 layer（仅 RGB 或仅 Depth）

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
        # custom_forward 只接收需要梯度的 tensor
        # attention_mask、rotary_pos_emb 通过闭包捕获
        def custom_forward(states: torch.Tensor) -> torch.Tensor:
            return layer(
                states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                output_attentions=False,  # checkpointing 时必须关闭
            )[0]

        hidden_states = self._gradient_checkpointing_func(custom_forward, hidden_states)
        return hidden_states, None  # 不返回 attention weights

    layer_outputs = layer(
        hidden_states,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        output_attentions=output_attentions,
    )
    return layer_outputs[0], layer_outputs[1] if output_attentions else None
```

### 双分支 mixture layer（RGB + Depth 联合处理）

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
        # 多个需要梯度的输入都作为显式参数传入
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

## Step 2（续）: 在 Encoder.forward 中调用

```python
def forward(self, hidden_states, hidden_states_depth=None, ...):
    use_gradient_checkpointing = self.gradient_checkpointing and self.training and not output_attentions

    for layer_i, layer in enumerate(self.layers):
        if layer_i in self._mot_layer_to_index:
            # MoT 混合层
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
            # 单分支 RGB layer
            hidden_states, attn = self._forward_single_branch_layer(
                layer, hidden_states,
                attention_mask, rotary_pos_emb,
                output_attentions, use_gradient_checkpointing,
            )
            # 单分支 Depth layer
            if hidden_states_depth is not None:
                depth_layer = self.layers_depth[layer_i]
                hidden_states_depth, attn_depth = self._forward_single_branch_layer(
                    depth_layer, hidden_states_depth,
                    attention_mask_depth, rotary_pos_emb_depth,
                    output_attentions, use_gradient_checkpointing,
                )
    # ...
```

## Step 3: Model 顶层（HuggingFace 方式）

```python
class TomatoViTPreTrainedModel(PreTrainedModel):
    config_class = TomatoViTConfig
    supports_gradient_checkpointing = True  # 这一行即可
```

`PreTrainedModel.gradient_checkpointing_enable()` 会自动找到 Encoder 并设置状态。

## 在 Engine 中启用

```python
# recipes/tomatovit/engine/tomatovit_engine.py — prepare_model() 内
gc_enabled = OmegaConf.select(self.config, "model.gradient_checkpointing.enabled", default=False)
gc_use_reentrant = OmegaConf.select(self.config, "model.gradient_checkpointing.use_reentrant", default=False)
if gc_enabled:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc_use_reentrant}
    )
```

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| checkpointing 粒度 | 每层一次 | 平衡显存节省与重计算开销 |
| `custom_forward` 参数 | 只放需要梯度的 tensor | `checkpoint()` 只对显式参数做 save/recompute |
| `output_attentions` | checkpointing 时强制 False | attention weights 无法在 recompute 时恢复 |
| 多种 layer 类型 | 各写一个 helper | 逻辑清晰，不过度抽象 |
| `use_reentrant` | 默认 False | 兼容 `torch.compile` 与非确定性操作 |
