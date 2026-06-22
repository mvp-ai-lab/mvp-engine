"""Qwen3 text-model helpers for the pretrain stage."""

from __future__ import annotations

from types import MethodType

import torch


def calculate_model_flops(
    self,
    *,
    batch_size: int,
    seq_len: int,
    attention_mask: torch.Tensor | None = None,
    is_training: bool = True,
    freeze_llm: bool = False,
) -> float:
    """Estimate local-rank logical Qwen3 text FLOPs for one prepared batch."""
    batch = int(batch_size)
    tokens = int(seq_len)
    if batch <= 0 or tokens <= 0:
        raise ValueError("batch_size and seq_len must be > 0")

    config = self.config  # flat Qwen3 text config (not config.text_config)
    layers = int(config.num_hidden_layers)
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    vocab = int(config.vocab_size)

    attention_token_pairs = _count_attention_token_pairs(
        batch=batch,
        tokens=tokens,
        attention_mask=attention_mask,
    )

    per_layer = (
        8 * batch * tokens * hidden * hidden
        + 4 * attention_token_pairs * hidden
        + 6 * batch * tokens * hidden * intermediate
    )
    flops = float(layers * per_layer + 2 * batch * tokens * hidden * vocab)

    if not is_training:
        return flops
    # Forward + backward is ~3x forward; a frozen LM only does the forward pass.
    return flops * (1.0 if freeze_llm else 3.0)


def patch_qwen3_model_flops(model):
    """Inject Qwen3 text FLOPs estimation into the runtime model instance."""
    model.calculate_model_flops = MethodType(calculate_model_flops, model)
    return model


def _count_attention_token_pairs(
    *,
    batch: int,
    tokens: int,
    attention_mask: torch.Tensor | None,
) -> int:
    """Count logical attention token pairs for padded or packed text tokens."""
    attention_token_pairs = batch * tokens * tokens
    if attention_mask is None or attention_mask.ndim != 2:
        return int(attention_token_pairs)

    mask = attention_mask.detach().to(device="cpu", dtype=torch.long)
    max_mask_value = int(mask.max().item()) if mask.numel() > 0 else 0
    if max_mask_value > 1:
        # Packed segment ids: each document attends only within itself.
        attention_token_pairs = 0
        for row in mask:
            segment_ids = row[row > 0]
            if segment_ids.numel() == 0:
                continue
            segment_lengths = torch.bincount(segment_ids, minlength=max_mask_value + 1)[1:]
            attention_token_pairs += int(torch.square(segment_lengths).sum().item())
    else:
        # A 0/1 mask is a padding mask (or a single packed segment): padded
        # tokens never participate in attention, so count only valid lengths.
        # This is the logical attention pattern for sdpa/eager and FA2 alike.
        valid_lengths = mask.ne(0).sum(dim=-1)
        attention_token_pairs = int(torch.square(valid_lengths).sum().item())

    if attention_token_pairs <= 0:
        raise ValueError("attention_mask must contain at least one valid token")
    return int(attention_token_pairs)
