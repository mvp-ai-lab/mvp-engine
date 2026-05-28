"""Qwen3-VL packed position-id helpers."""

from typing import Any

import torch


def build_qwen3_vl_packed_position_ids(
    *,
    input_ids: torch.Tensor,
    pack_segment_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    model_config: Any,
) -> torch.Tensor:
    """Build packed Qwen3-VL RoPE position ids with cumulative packed offsets."""
    if input_ids.ndim != 2:
        raise ValueError(f"Expected 2D input_ids, got shape {tuple(input_ids.shape)}.")
    if pack_segment_ids.shape != input_ids.shape:
        raise ValueError(
            "pack_segment_ids must have the same shape as input_ids, "
            f"got {tuple(pack_segment_ids.shape)} vs {tuple(input_ids.shape)}."
        )

    device = input_ids.device
    batch_size, sequence_length = input_ids.shape
    position_ids = torch.ones((3, batch_size, sequence_length), dtype=torch.long, device=device)
    image_cursor = 0

    for batch_index in range(batch_size):
        segment_ids = pack_segment_ids[batch_index]
        sample_tokens = input_ids[batch_index]
        next_position = 0
        for segment_id in segment_ids[segment_ids > 0].unique_consecutive().tolist():
            segment_mask = segment_ids == segment_id
            segment_tokens = sample_tokens[segment_mask]
            vision_start_indices = torch.argwhere(segment_tokens == int(model_config.vision_start_token_id)).squeeze(1)
            vision_start_indices = vision_start_indices[vision_start_indices + 1 < segment_tokens.shape[0]]
            segment_image_count = 0
            if vision_start_indices.numel() > 0:
                vision_tokens = segment_tokens[vision_start_indices + 1]
                segment_image_count = int((vision_tokens == int(model_config.image_token_id)).sum().item())
            segment_grids = None
            if segment_image_count > 0:
                if image_grid_thw is None:
                    raise ValueError("Packed multimodal samples require image_grid_thw to build position_ids.")
                segment_grids = image_grid_thw[image_cursor : image_cursor + segment_image_count]
                if segment_grids.shape[0] != segment_image_count:
                    raise ValueError("image_grid_thw does not match the number of packed image samples.")
                image_cursor += segment_image_count

            segment_position_ids = _build_qwen3_vl_segment_position_ids(
                input_ids=segment_tokens,
                image_grid_thw=segment_grids,
                model_config=model_config,
            )
            segment_position_ids = segment_position_ids + next_position
            position_ids[:, batch_index, segment_mask] = segment_position_ids
            next_position = int(segment_position_ids.max().item()) + 1

    if image_grid_thw is not None and image_cursor != int(image_grid_thw.shape[0]):
        raise ValueError(
            "Packed image metadata does not align with packed token segments: "
            f"consumed {image_cursor}, available {int(image_grid_thw.shape[0])}."
        )

    return position_ids


def _build_qwen3_vl_segment_position_ids(
    *,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    model_config: Any,
) -> torch.Tensor:
    """Mirror Qwen3-VL rope indexing for one packed segment."""
    spatial_merge_size = int(model_config.vision_config.spatial_merge_size)
    image_token_id = int(model_config.image_token_id)
    video_token_id = int(model_config.video_token_id)
    vision_start_token_id = int(model_config.vision_start_token_id)

    if bool((input_ids == video_token_id).any().item()):
        raise NotImplementedError("Packed Basic VLM training only supports text+image inputs, not video inputs.")

    vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
    if vision_start_indices.numel() == 0:
        return _build_text_position_ids(input_ids)

    vision_start_indices = vision_start_indices[vision_start_indices + 1 < input_ids.shape[0]]
    vision_tokens = input_ids[vision_start_indices + 1]
    image_count = int((vision_tokens == image_token_id).sum().item())
    if image_count == 0:
        return _build_text_position_ids(input_ids)
    if image_grid_thw is None or int(image_grid_thw.shape[0]) != image_count:
        grid_count = 0 if image_grid_thw is None else int(image_grid_thw.shape[0])
        raise ValueError(
            "Packed Qwen3-VL segment image metadata mismatch: "
            f"expected {image_count} image grid rows, got {grid_count}."
        )

    image_token_positions = (vision_start_indices + 1).tolist()
    llm_position_chunks: list[torch.Tensor] = []
    segment_start = 0

    for image_index in range(image_count):
        image_start = image_token_positions[image_index]
        t, h, w = image_grid_thw[image_index]

        llm_grid_t = int(t.item())
        llm_grid_h = int(h.item()) // spatial_merge_size
        llm_grid_w = int(w.item()) // spatial_merge_size
        text_length = image_start - segment_start
        next_position = int(llm_position_chunks[-1].max().item()) + 1 if llm_position_chunks else 0

        if text_length > 0:
            llm_position_chunks.append(
                torch.arange(text_length, device=input_ids.device, dtype=torch.long).view(1, -1).expand(3, -1)
                + next_position
            )
            next_position += text_length

        t_index = torch.arange(llm_grid_t, device=input_ids.device, dtype=torch.long).view(-1, 1)
        t_index = t_index.expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h, device=input_ids.device, dtype=torch.long).view(1, -1, 1)
        h_index = h_index.expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w, device=input_ids.device, dtype=torch.long).view(1, 1, -1)
        w_index = w_index.expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_position_chunks.append(torch.stack([t_index, h_index, w_index]) + next_position)
        segment_start = image_start + (llm_grid_t * llm_grid_h * llm_grid_w)

    if segment_start < input_ids.shape[0]:
        next_position = int(llm_position_chunks[-1].max().item()) + 1 if llm_position_chunks else 0
        tail_length = input_ids.shape[0] - segment_start
        llm_position_chunks.append(
            torch.arange(tail_length, device=input_ids.device, dtype=torch.long).view(1, -1).expand(3, -1)
            + next_position
        )

    return torch.cat(llm_position_chunks, dim=1)


def _build_text_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Build text-only 3D position ids for one Qwen3-VL segment."""
    return torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long).view(1, -1).expand(3, -1)


__all__ = ["build_qwen3_vl_packed_position_ids"]
