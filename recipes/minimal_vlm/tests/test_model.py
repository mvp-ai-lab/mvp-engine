from __future__ import annotations

import torch.nn as nn

from recipes.minimal_vlm.model import freeze_visual_parameters


class FakeQwen3VLModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.model.language_model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.lm_head = nn.Linear(4, 8)


def test_freeze_visual_parameters_only_freezes_visual_subtree() -> None:
    model = FakeQwen3VLModel()

    freeze_visual_parameters(model, freeze_visual=True)

    assert all(not parameter.requires_grad for parameter in model.model.visual.parameters())
    assert all(parameter.requires_grad for parameter in model.model.language_model.parameters())
    assert all(parameter.requires_grad for parameter in model.lm_head.parameters())
