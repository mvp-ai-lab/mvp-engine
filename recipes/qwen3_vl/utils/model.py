import torch.nn as nn


def freeze_module(
    module: nn.Module,
    include_keywords: tuple[str, ...] = (),
    exclude_keywords: tuple[str, ...] = (),
) -> int:
    """Freeze parameters in a module with optional include/exclude keyword filters."""
    freeze = 0
    if include_keywords or exclude_keywords:
        for name, param in module.named_parameters():
            if include_keywords and not any(keyword in name for keyword in include_keywords):
                continue
            if any(keyword in name for keyword in exclude_keywords):
                continue
            if param.requires_grad:
                param.requires_grad = False
                freeze += param.numel()
    else:
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False
                freeze += param.numel()
    return freeze
