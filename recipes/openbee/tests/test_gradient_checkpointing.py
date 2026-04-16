import importlib
import sys
import types
from types import SimpleNamespace


def _install_test_stubs() -> None:
    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.__path__ = []

        class AutoModelForImageTextToText:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise NotImplementedError

        transformers.AutoModelForImageTextToText = AutoModelForImageTextToText
        sys.modules["transformers"] = transformers

        transformers_models = types.ModuleType("transformers.models")
        transformers_models.__path__ = []
        sys.modules["transformers.models"] = transformers_models

        qwen3_vl_pkg = types.ModuleType("transformers.models.qwen3_vl")
        qwen3_vl_pkg.__path__ = []
        sys.modules["transformers.models.qwen3_vl"] = qwen3_vl_pkg

        modeling_qwen3_vl = types.ModuleType("transformers.models.qwen3_vl.modeling_qwen3_vl")

        class Qwen3VLCausalLMOutputWithPast:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        modeling_qwen3_vl.Qwen3VLCausalLMOutputWithPast = Qwen3VLCausalLMOutputWithPast
        sys.modules["transformers.models.qwen3_vl.modeling_qwen3_vl"] = modeling_qwen3_vl


_install_test_stubs()
qwen3_vl = importlib.import_module("recipes.openbee.model.qwen3_vl")


class _FakeCheckpointLayer:
    def __init__(self):
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None


class _FakeModel:
    def __init__(self):
        self.model = SimpleNamespace(
            visual=SimpleNamespace(
                patch_embed=SimpleNamespace(),
                blocks=[_FakeCheckpointLayer()],
            ),
            language_model=SimpleNamespace(layers=[_FakeCheckpointLayer()]),
        )
        self.enable_calls: list[dict[str, bool] | None] = []

    def named_parameters(self):
        return []

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.enable_calls.append(gradient_checkpointing_kwargs)
        for layer in self.model.visual.blocks + self.model.language_model.layers:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = lambda fn, *args, **kwargs: fn(*args, **kwargs)


def _build_model_config(*, enabled: bool, use_reentrant: bool):
    return SimpleNamespace(
        pretrained_model_name_or_path="/tmp/qwen3-vl",
        attn_implementation="flash_attention_2",
        gradient_checkpointing=SimpleNamespace(
            enabled=enabled,
            use_reentrant=use_reentrant,
        ),
        freeze_vit=False,
        freeze_merger=False,
        freeze_llm=False,
    )


def test_build_qwen3_vl_model_enables_gradient_checkpointing(monkeypatch):
    fake_model = _FakeModel()
    monkeypatch.setattr(
        qwen3_vl,
        "AutoModelForImageTextToText",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake_model),
    )

    model = qwen3_vl.build_qwen3_vl_model(
        _build_model_config(enabled=True, use_reentrant=False),
    )

    assert model is fake_model
    assert fake_model.enable_calls == [{"use_reentrant": False}]
    assert fake_model.model.visual.blocks[0].gradient_checkpointing is True
    assert fake_model.model.language_model.layers[0].gradient_checkpointing is True
    assert callable(fake_model.model.visual.blocks[0]._gradient_checkpointing_func)


def test_build_qwen3_vl_model_skips_gradient_checkpointing_when_disabled(monkeypatch):
    fake_model = _FakeModel()
    monkeypatch.setattr(
        qwen3_vl,
        "AutoModelForImageTextToText",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake_model),
    )

    model = qwen3_vl.build_qwen3_vl_model(
        _build_model_config(enabled=False, use_reentrant=False),
    )

    assert model is fake_model
    assert fake_model.enable_calls == []
    assert fake_model.model.visual.blocks[0].gradient_checkpointing is False
    assert fake_model.model.language_model.layers[0].gradient_checkpointing is False
