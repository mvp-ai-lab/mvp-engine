# Gradient Checkpointing 测试模板

为每个模型编写 3 个测试。以下为可复制模板，将 `{Model}`、`{Encoder}`、`{Config}` 替换为实际类型即可。  
**English:** [test-patterns.md](test-patterns.md)

## 测试 1: enable/disable 正确设置状态

验证 `gradient_checkpointing_enable()` 与 `gradient_checkpointing_disable()` 正确设置 encoder 属性。

```python
def test_gradient_checkpointing_enable_sets_state():
    config = {Config}(...)  # 最小配置
    model = {Model}(config)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.encoder.gradient_checkpointing is True
    assert callable(model.encoder._gradient_checkpointing_func)

    model.gradient_checkpointing_disable()
    assert model.encoder.gradient_checkpointing is False
```

对纯 `nn.Module` 模型（非 HuggingFace），直接调用自定义的 enable/disable 方法。

## 测试 2: 训练时实际调用了 checkpoint 函数

用 fake checkpoint 函数记录调用次数，确认每一层都被 checkpoint 包裹。

```python
def test_encoder_uses_checkpointing():
    config = {Config}(...)
    encoder = {Encoder}(config)

    # 用 DummyLayer 替换真实 layer 以加速测试
    encoder.layers = nn.ModuleList([DummyLayer() for _ in range(num_layers)])

    checkpoint_calls = []

    def fake_gc(func, *args, **kwargs):
        checkpoint_calls.append(func.__name__)
        return func(*args, **kwargs)

    encoder.gradient_checkpointing = True
    encoder._gradient_checkpointing_func = fake_gc
    encoder.train()

    hidden_states = torch.randn(2, 4, hidden_size, requires_grad=True)
    encoder(hidden_states=hidden_states, output_attentions=False, return_dict=True)

    assert len(checkpoint_calls) == num_layers  # 每层调用一次
```

### DummyLayer 要求

`DummyLayer.forward` 的签名必须与真实 layer 完全一致：

```python
class DummyLayer(nn.Module):
    """与真实 layer 签名一致的 dummy，用于测试 checkpointing 包装逻辑。"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        rotary_pos_emb=None,
        output_attentions: bool = False,
    ):
        next_states = hidden_states + 1  # 简单可验证的变换
        if output_attentions:
            return (next_states, torch.zeros(1, device=hidden_states.device))
        return (next_states,)
```

若模型有多种 layer 类型，为每种类型各写一个 DummyLayer 和对应测试。

## 测试 3: 梯度数值一致性

最重要的一项：开启 checkpointing 后的梯度应与关闭时一致。

```python
def test_gradient_matches_without_checkpointing():
    config = {Config}(...)
    torch.manual_seed(42)

    # 不开启 checkpointing
    model_ref = {Model}(config)
    model_ref.train()
    x = torch.randn(1, seq_len, hidden_size, requires_grad=True)
    out_ref = model_ref(x).last_hidden_state.sum()
    out_ref.backward()
    grad_ref = {收集关键参数的梯度}

    # 开启 checkpointing
    torch.manual_seed(42)
    model_gc = {Model}(config)
    model_gc.load_state_dict(model_ref.state_dict())  # 确保权重一致
    model_gc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model_gc.train()
    x_gc = x.detach().clone().requires_grad_(True)
    out_gc = model_gc(x_gc).last_hidden_state.sum()
    out_gc.backward()
    grad_gc = {收集关键参数的梯度}

    # 验证梯度一致
    for g_ref, g_gc in zip(grad_ref, grad_gc):
        assert torch.allclose(g_ref, g_gc, atol=1e-5), "Gradient mismatch with checkpointing enabled"
```

## TomatoViT 完整测试参考

仓库中的实际测试文件：`tests/test_tomatovit_gradient_checkpointing.py`

要点：
- 用 `pytest.importorskip("flash_attn")` 在缺少依赖时跳过。
- `_tiny_config()` 创建最小配置（如 `hidden_size=32`、`patch_size=16`），保证测试秒级完成。
- 分别测试 regular layer 与 mixture layer 的 checkpointing。
- 用 `checkpoint_calls` 列表验证调用次数与目标函数名。
- 用 `torch.allclose` 验证输出数值。

## 测试命名规范

```
tests/test_{model_name}_gradient_checkpointing.py
```

函数命名建议：
- `test_gradient_checkpointing_enable_sets_{encoder_name}_state`
- `test_{encoder_name}_uses_checkpointing_for_{layer_type}_layers`
- `test_gradient_matches_without_checkpointing`
