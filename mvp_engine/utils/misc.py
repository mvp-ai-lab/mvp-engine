import subprocess

import torch
import torch.nn as nn


def freeze(module: nn.Module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def find_optimizable_params(module: nn.Module):
    for p in module.parameters():
        if p.requires_grad:
            yield p


def get_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        try:
            import torch_npu  # noqa: F401

            return torch.device(f"npu:{index}")
        except ImportError:
            return torch.device("cpu")


def get_git_info():
    branch = "None"
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.STDOUT,
            )
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError:
        pass

    commit_hash = "None"
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode("utf-8").strip()
        )
    except subprocess.CalledProcessError:
        pass

    return {"branch": branch, "commit_hash": commit_hash}


def calculate_model_size(model: nn.Module):
    """Log total and trainable parameter counts for a model.

    This utility computes the total number of parameters and the number of
    trainable parameters in the given model in distributed training setups (e.g., DDP/FSDP2)
    and logs both in billions of parameters.

    Args:
        model: A PyTorch ``nn.Module`` (or compatible object) whose
            parameters will be counted.
    """
    model_size = sum(p.numel() for p in model.parameters())
    trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model_size, trainable_size
