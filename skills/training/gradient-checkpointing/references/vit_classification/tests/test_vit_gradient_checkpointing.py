import copy

import pytest
import torch

pytest.importorskip("transformers")

from recipes.vit_classification.configs.schema import ViTModelConfig
from recipes.vit_classification.model.vit import build_vit_model


def _build_test_config():
    return ViTModelConfig(
        pretrained_model_name_or_path="google/vit-base-patch16-224",
        load_pretrained_weights=False,
        num_classes=3,
        image_size=16,
        patch_size=8,
        num_channels=3,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
    )


def test_gradient_checkpointing_enable_sets_vit_layer_state():
    model = build_vit_model(_build_test_config())

    assert not model.is_gradient_checkpointing

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    assert model.is_gradient_checkpointing
    assert model.vit.encoder.layer[0].gradient_checkpointing is True
    assert callable(model.vit.encoder.layer[0]._gradient_checkpointing_func)

    model.gradient_checkpointing_disable()

    assert not model.is_gradient_checkpointing
    assert model.vit.encoder.layer[0].gradient_checkpointing is False


def test_vit_layers_use_checkpoint_function_during_training():
    model = build_vit_model(_build_test_config())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    calls = []

    def fake_checkpoint(function, *args):
        calls.append(len(args))
        return function(*args)

    for layer in model.vit.encoder.layer:
        layer._gradient_checkpointing_func = fake_checkpoint

    outputs = model(
        pixel_values=torch.randn(2, 3, 16, 16),
        labels=torch.tensor([0, 1]),
    )
    outputs.loss.backward()

    assert len(calls) == model.config.num_hidden_layers


def test_vit_gradients_match_with_and_without_checkpointing():
    config = _build_test_config()
    baseline_model = build_vit_model(config)
    checkpointed_model = build_vit_model(config)
    checkpointed_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))
    checkpointed_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    pixel_values = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([0, 1])

    baseline_model.train()
    baseline_outputs = baseline_model(pixel_values=pixel_values, labels=labels)
    baseline_outputs.loss.backward()

    checkpointed_model.train()
    checkpointed_outputs = checkpointed_model(pixel_values=pixel_values, labels=labels)
    checkpointed_outputs.loss.backward()

    baseline_grads = dict(baseline_model.named_parameters())
    checkpointed_grads = dict(checkpointed_model.named_parameters())
    assert baseline_grads.keys() == checkpointed_grads.keys()

    for name in baseline_grads:
        baseline_grad = baseline_grads[name].grad
        checkpointed_grad = checkpointed_grads[name].grad
        assert baseline_grad is not None, name
        assert checkpointed_grad is not None, name
        torch.testing.assert_close(baseline_grad, checkpointed_grad, rtol=1e-5, atol=1e-6)
