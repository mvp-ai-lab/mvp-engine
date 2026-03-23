from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from recipes.minimal_vlm.model import apply_freeze_policy, build_qwen3_vl_model
from recipes.minimal_vlm.model import qwen3_vl as qwen3_vl_module


class FakeQwen3VLModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.model.visual.merger = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.model.visual.deepstack_merger_list = nn.ModuleList([nn.Linear(4, 4)])
        self.model.language_model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.lm_head = nn.Linear(4, 8)


def test_apply_freeze_policy_only_freezes_visual_groups_by_default_policy() -> None:
    model = FakeQwen3VLModel()

    apply_freeze_policy(model, freeze_vit=True, freeze_projector=True, freeze_llm=False)

    assert all(not parameter.requires_grad for parameter in model.model.visual.blocks.parameters())
    assert all(not parameter.requires_grad for parameter in model.model.visual.merger.parameters())
    assert all(not parameter.requires_grad for parameter in model.model.visual.deepstack_merger_list.parameters())
    assert all(parameter.requires_grad for parameter in model.model.language_model.parameters())
    assert all(parameter.requires_grad for parameter in model.lm_head.parameters())


def test_apply_freeze_policy_can_freeze_only_projector_and_llm() -> None:
    model = FakeQwen3VLModel()

    apply_freeze_policy(model, freeze_vit=False, freeze_projector=True, freeze_llm=True)

    assert all(parameter.requires_grad for parameter in model.model.visual.blocks.parameters())
    assert all(not parameter.requires_grad for parameter in model.model.visual.merger.parameters())
    assert all(not parameter.requires_grad for parameter in model.model.visual.deepstack_merger_list.parameters())
    assert all(not parameter.requires_grad for parameter in model.model.language_model.parameters())
    assert all(not parameter.requires_grad for parameter in model.lm_head.parameters())


def test_build_qwen3_vl_model_passes_attention_implementation(monkeypatch) -> None:
    captured_kwargs = {}
    fake_model = FakeQwen3VLModel()
    fake_model.config = SimpleNamespace(_attn_implementation=None)

    def fake_from_pretrained(*args, **kwargs):
        del args
        captured_kwargs.update(kwargs)
        return fake_model

    monkeypatch.setattr(qwen3_vl_module.AutoModelForImageTextToText, "from_pretrained", fake_from_pretrained)

    model = build_qwen3_vl_model(
        SimpleNamespace(
            pretrained_model_name_or_path="dummy",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            freeze_vit=False,
            freeze_projector=False,
            freeze_llm=False,
        )
    )

    assert model is fake_model
    assert captured_kwargs["attn_implementation"] == "flash_attention_2"
    assert model.config._attn_implementation == "flash_attention_2"
