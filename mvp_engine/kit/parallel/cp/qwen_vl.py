"""Qwen-VL context-parallel visual helpers."""

from __future__ import annotations

import torch

from .cp import CPKit


class QwenVLCPKit(CPKit):
    """Qwen-VL visual helpers plus the generic dense-sequence CPKit API."""

    def local_visual_patch_indices(
        self,
        grid_thw: torch.Tensor,
        *,
        pad_scale: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return local flattened patch indices aligned to Qwen visual merge groups."""
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError(f"grid_thw must have shape [num_items, 3], got {tuple(grid_thw.shape)}.")
        pad_scale = int(pad_scale)
        if pad_scale <= 0:
            raise ValueError(f"pad_scale must be positive, got {pad_scale}.")

        total_patches = int(grid_thw.to(dtype=torch.long).prod(dim=-1).sum().item())
        if total_patches == 0:
            return torch.empty(0, device=device, dtype=torch.long)
        if total_patches % pad_scale != 0:
            raise ValueError(f"Qwen visual patch count ({total_patches}) must be divisible by pad_scale ({pad_scale}).")

        merged_tokens = total_patches // pad_scale
        if self.context_size <= 1:
            local_merged_indices = torch.arange(merged_tokens, device=device)
        else:
            padded_merged_tokens = ((merged_tokens + self.context_size - 1) // self.context_size) * self.context_size
            local_merged_indices = torch.arange(padded_merged_tokens, device=device).chunk(self.context_size, dim=0)[
                self.context_rank
            ]
        offsets = torch.arange(pad_scale, device=local_merged_indices.device)
        return (local_merged_indices[:, None] * pad_scale + offsets[None, :]).reshape(-1).clamp(max=total_patches - 1)
