# Gradient Checkpointing Test Templates

Write 3 tests per model. Below are copy-paste templates; replace `{Model}`, `{Encoder}`, `{Config}` as needed.  
**中文：** [test-patterns.zh-CN.md](test-patterns.zh-CN.md)

## Test 1: enable/disable sets state correctly

Assert that `gradient_checkpointing_enable()` and `gradient_checkpointing_disable()` set encoder attributes correctly.

```python
def test_gradient_checkpointing_enable_sets_state():
    config = {Config}(...)  # minimal config
    model = {Model}(config)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.encoder.gradient_checkpointing is True
    assert callable(model.encoder._gradient_checkpointing_func)

    model.gradient_checkpointing_disable()
    assert model.encoder.gradient_checkpointing is False
```

For plain `nn.Module` models (non-HuggingFace), call your custom enable/disable methods.

## Test 2: Checkpoint function is used during training

Use a fake checkpoint function that records calls to confirm each layer is wrapped.

```python
def test_encoder_uses_checkpointing():
    config = {Config}(...)
    encoder = {Encoder}(config)

    # Replace real layers with DummyLayer for fast tests
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

    assert len(checkpoint_calls) == num_layers  # one call per layer
```

### DummyLayer requirements

`DummyLayer.forward` must match the real layer’s signature exactly:

```python
class DummyLayer(nn.Module):
    """Dummy with same signature as real layer, for testing checkpointing wrapper."""
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        rotary_pos_emb=None,
        output_attentions: bool = False,
    ):
        next_states = hidden_states + 1  # simple, verifiable transform
        if output_attentions:
            return (next_states, torch.zeros(1, device=hidden_states.device))
        return (next_states,)
```

If the model has multiple layer types, define a DummyLayer and test per type.

## Test 3: Gradient numerical consistency

Most important test — gradients with checkpointing on must match those with it off. Reseed before each forward so both branches consume RNG identically; otherwise dropout (or other stochastic ops) can make the test fail even when checkpointing is correct.

```python
def test_gradient_matches_without_checkpointing():
    config = {Config}(...)

    # Without checkpointing
    torch.manual_seed(42)
    model_ref = {Model}(config)
    model_ref.train()
    x = torch.randn(1, seq_len, hidden_size, requires_grad=True)
    out_ref = model_ref(x).last_hidden_state.sum()
    out_ref.backward()
    grad_ref = {collect gradients of key params}

    # With checkpointing — reseed so RNG state matches before forward (avoids false negatives with dropout)
    torch.manual_seed(42)
    model_gc = {Model}(config)
    model_gc.load_state_dict(model_ref.state_dict())  # same weights
    model_gc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model_gc.train()
    x_gc = torch.randn(1, seq_len, hidden_size, requires_grad=True)  # fresh sample after reseed
    out_gc = model_gc(x_gc).last_hidden_state.sum()
    out_gc.backward()
    grad_gc = {collect gradients of key params}

    # Assert gradients match
    for g_ref, g_gc in zip(grad_ref, grad_gc):
        assert torch.allclose(g_ref, g_gc, atol=1e-5), "Gradient mismatch with checkpointing enabled"
```

## TomatoViT full test reference

Actual test file in the repo: `tests/test_tomatovit_gradient_checkpointing.py`

Notes:
- Use `pytest.importorskip("flash_attn")` to skip when the dependency is missing.
- `_tiny_config()` builds a minimal config (`hidden_size=32`, `patch_size=16`, etc.) so tests finish in seconds.
- Test regular layers and mixture layers’ checkpointing separately.
- Use a `checkpoint_calls` list to assert call count and target function names.
- Use `torch.allclose` to assert output values.

## Naming convention

```
tests/test_{model_name}_gradient_checkpointing.py
```

Function names:
- `test_gradient_checkpointing_enable_sets_{encoder_name}_state`
- `test_{encoder_name}_uses_checkpointing_for_{layer_type}_layers`
- `test_gradient_matches_without_checkpointing`
