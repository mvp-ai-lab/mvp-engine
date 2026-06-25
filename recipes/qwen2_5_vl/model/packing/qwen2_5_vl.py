"""Image-only packed input preparation for qwen2_5_vl."""

from typing import Any

import torch

from mvp_engine.kit.mllm.data import build_packed_block_causal_mask


def prepare_packed_model_inputs(
    batch: dict[str, Any],
    *,
    model_config: Any,
    attn_implementation: str | None,
    mask_dtype: torch.dtype,
) -> dict[str, Any]:
    """Convert canonical packed metadata into model inputs."""
    _reject_unsupported_fields(batch, model_config)

    pack_segment_ids = batch.get("pack_segment_ids")
    if pack_segment_ids is None:
        raise ValueError("Packed qwen2_5_vl batches must include pack_segment_ids.")

    batch.pop("source_sample_num", None)
    batch.pop("num_input_tokens", None)
    batch.pop("num_loss_tokens", None)
    batch.pop("num_source_samples", None)

    batch["position_ids"] = build_qwen2_5_vl_packed_position_ids(
        input_ids=batch["input_ids"],
        pack_segment_ids=pack_segment_ids,
        image_grid_thw=batch.get("image_grid_thw"),
        model_config=model_config,
    )

    if attn_implementation == "flash_attention_2":
        batch["attention_mask"] = None
        batch.update(build_packed_fa2_varlen_kwargs(pack_segment_ids))
    else:
        batch["attention_mask"] = build_packed_block_causal_mask(pack_segment_ids, dtype=mask_dtype)

    return batch


def build_qwen2_5_vl_packed_position_ids(
    *,
    input_ids: torch.Tensor,
    pack_segment_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    model_config: Any,
) -> torch.Tensor:
    """Build packed Qwen2.5-VL text and mRoPE position ids."""
    if input_ids.ndim != 2:
        raise ValueError(f"Expected 2D input_ids, got shape {tuple(input_ids.shape)}.")
    if pack_segment_ids.shape != input_ids.shape:
        raise ValueError(
            "pack_segment_ids must have the same shape as input_ids, "
            f"got {tuple(pack_segment_ids.shape)} vs {tuple(input_ids.shape)}."
        )

    position_ids = torch.zeros((4, *input_ids.shape), dtype=torch.long, device=input_ids.device)
    image_cursor = 0
    for batch_index in range(input_ids.shape[0]):
        row_segments = pack_segment_ids[batch_index]
        row_tokens = input_ids[batch_index]
        for segment_id in row_segments[row_segments > 0].unique_consecutive().tolist():
            segment_mask = row_segments == segment_id
            segment_tokens = row_tokens[segment_mask]
            segment_positions, used_images = _build_segment_position_ids(
                input_ids=segment_tokens,
                image_grid_thw=None if image_grid_thw is None else image_grid_thw[image_cursor:],
                model_config=model_config,
            )
            position_ids[:, batch_index, segment_mask] = segment_positions
            image_cursor += used_images

    if image_grid_thw is not None and image_cursor != int(image_grid_thw.shape[0]):
        raise ValueError(
            "Packed image metadata does not align with packed token segments: "
            f"consumed {image_cursor}, available {int(image_grid_thw.shape[0])}."
        )
    return position_ids


def build_packed_fa2_varlen_kwargs(pack_segment_ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
    """Build FlashAttention varlen kwargs from packed segment ids."""
    if pack_segment_ids.ndim != 2:
        raise ValueError(f"Expected 2D pack_segment_ids, got shape {tuple(pack_segment_ids.shape)}.")

    segment_lengths = []
    for row in pack_segment_ids:
        valid_length = int(row.ne(0).sum().item())
        if valid_length <= 0:
            raise ValueError("Each packed FlashAttention row must contain at least one non-padding token.")
        if bool(row[:valid_length].eq(0).any().item()) or bool(row[valid_length:].ne(0).any().item()):
            raise ValueError("Packed FlashAttention padding must be a single zero-valued suffix.")

        valid_row = row[:valid_length]
        starts = torch.cat(
            [
                torch.zeros(1, device=row.device, dtype=torch.long),
                torch.nonzero(valid_row[1:] != valid_row[:-1], as_tuple=False).flatten() + 1,
            ]
        )
        ends = torch.cat([starts[1:], torch.tensor([valid_length], device=row.device, dtype=torch.long)])
        segment_lengths.append(ends - starts)

    seqlens = torch.cat(segment_lengths).to(dtype=torch.int32)
    cu_seqlens = torch.zeros(seqlens.numel() + 1, device=pack_segment_ids.device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    return {
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": int(seqlens.max().item()),
        "max_length_k": int(seqlens.max().item()),
    }


def _build_segment_position_ids(
    *,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    model_config: Any,
) -> tuple[torch.Tensor, int]:
    image_token_id = int(model_config.image_token_id)
    spatial_merge_size = int(model_config.vision_config.spatial_merge_size)
    text_position_ids = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
    vision_position_ids = text_position_ids.view(1, -1).expand(3, -1).clone()

    image_spans = _find_contiguous_token_spans(input_ids, image_token_id)
    if not image_spans:
        return torch.cat([text_position_ids.view(1, -1), vision_position_ids], dim=0), 0
    if image_grid_thw is None or int(image_grid_thw.shape[0]) < len(image_spans):
        available = 0 if image_grid_thw is None else int(image_grid_thw.shape[0])
        raise ValueError(f"Packed image metadata mismatch: expected {len(image_spans)} rows, got {available}.")

    current_position = 0
    consumed_images = 0
    previous_end = 0
    for image_index, (image_start, image_end) in enumerate(image_spans):
        if image_start > previous_end:
            vision_position_ids[:, previous_end:image_start] = torch.arange(
                current_position,
                current_position + image_start - previous_end,
                device=input_ids.device,
                dtype=torch.long,
            ).view(1, -1)
            current_position += image_start - previous_end

        grid = image_grid_thw[image_index]
        llm_grid_t = int(grid[0].item())
        llm_grid_h = int(grid[1].item()) // spatial_merge_size
        llm_grid_w = int(grid[2].item()) // spatial_merge_size
        expected_tokens = llm_grid_t * llm_grid_h * llm_grid_w
        actual_tokens = image_end - image_start
        if actual_tokens != expected_tokens:
            raise ValueError(
                f"Image token span does not match image_grid_thw: expected {expected_tokens}, got {actual_tokens}."
            )

        t_index = torch.arange(llm_grid_t, device=input_ids.device, dtype=torch.long).view(-1, 1)
        t_index = t_index.expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h, device=input_ids.device, dtype=torch.long).view(1, -1, 1)
        h_index = h_index.expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w, device=input_ids.device, dtype=torch.long).view(1, 1, -1)
        w_index = w_index.expand(llm_grid_t, llm_grid_h, -1).flatten()
        vision_position_ids[:, image_start:image_end] = torch.stack([t_index, h_index, w_index]) + current_position
        current_position += max(llm_grid_t, llm_grid_h, llm_grid_w)
        previous_end = image_end
        consumed_images += 1

    if previous_end < input_ids.shape[0]:
        vision_position_ids[:, previous_end:] = torch.arange(
            current_position,
            current_position + input_ids.shape[0] - previous_end,
            device=input_ids.device,
            dtype=torch.long,
        ).view(1, -1)

    return torch.cat([text_position_ids.view(1, -1), vision_position_ids], dim=0), consumed_images


def _find_contiguous_token_spans(input_ids: torch.Tensor, token_id: int) -> list[tuple[int, int]]:
    matches = torch.nonzero(input_ids == token_id, as_tuple=False).flatten()
    if matches.numel() == 0:
        return []

    spans = []
    start = int(matches[0].item())
    previous = start
    for index in matches[1:].tolist():
        index = int(index)
        if index != previous + 1:
            spans.append((start, previous + 1))
            start = index
        previous = index
    spans.append((start, previous + 1))
    return spans


def _reject_unsupported_fields(batch: dict[str, Any], model_config: Any) -> None:
    unsupported_fields = ("video_grid_thw", "pixel_values_videos", "second_per_grid_ts")
    present_fields = [field for field in unsupported_fields if batch.get(field) is not None]
    if present_fields:
        raise NotImplementedError(f"qwen2_5_vl image-only recipe does not support fields: {present_fields}.")

    video_token_id = getattr(model_config, "video_token_id", None)
    if video_token_id is not None and "input_ids" in batch:
        if bool((batch["input_ids"] == int(video_token_id)).any().item()):
            raise NotImplementedError("qwen2_5_vl image-only recipe does not support video tokens.")


__all__ = [
    "build_packed_fa2_varlen_kwargs",
    "build_qwen2_5_vl_packed_position_ids",
    "prepare_packed_model_inputs",
]
