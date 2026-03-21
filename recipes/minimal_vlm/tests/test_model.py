from __future__ import annotations

import torch.nn as nn

from recipes.minimal_vlm.model import apply_freeze_policy


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
