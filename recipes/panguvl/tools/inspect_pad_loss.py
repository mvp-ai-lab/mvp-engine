"""Inspect whether PanguVL padding can enter supervised loss positions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from recipes.panguvl.configs.schema import PanguvlConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IGNORE_INDEX = -100


def _as_int(value: torch.Tensor) -> int:
    return int(value.detach().cpu().item())


def _count(mask: torch.Tensor) -> int:
    return int(mask.detach().sum().cpu().item())


def _shift_labels(labels: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    return F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


def _load_config(config_path: Path) -> PanguvlConfig:
    from omegaconf import OmegaConf

    from recipes.panguvl.configs.schema import PanguvlConfig

    raw_config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return PanguvlConfig.model_validate(raw_config)


def _print_pad_configuration(config: PanguvlConfig, processor: Any, model: torch.nn.Module | None) -> None:
    tokenizer = processor.tokenizer
    print("pad configuration:")
    print(f"  config.model.pad_token_id: {config.model.pad_token_id}")
    print(f"  tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    print(f"  tokenizer.pad_token: {tokenizer.pad_token!r}")

    if model is None:
        return

    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    embedding = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embedding is None:
        embedding = getattr(model, "embed_tokens", None)

    print(f"  model.config.pad_token_id: {getattr(model_config, 'pad_token_id', None)}")
    print(f"  model.config.text_config.pad_token_id: {getattr(text_config, 'pad_token_id', None)}")
    print(f"  embedding.padding_idx: {getattr(embedding, 'padding_idx', None)}")


def _decode_window(tokenizer: Any, input_ids: torch.Tensor, batch_index: int, position: int, *, radius: int = 8) -> str:
    start = max(0, position - radius)
    end = min(int(input_ids.shape[1]), position + radius + 1)
    token_ids = input_ids[batch_index, start:end].detach().cpu().tolist()
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return repr(token_ids)


def _inspect_batch(batch: dict[str, Any], *, pad_token_id: int, batch_index: int) -> dict[str, int]:
    import torch

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    shifted_labels = _shift_labels(labels)

    pad_positions = input_ids.eq(pad_token_id)
    ignored_labels = labels.eq(IGNORE_INDEX)
    supervised_labels = labels.ne(IGNORE_INDEX)
    shifted_supervised_labels = shifted_labels.ne(IGNORE_INDEX)

    stats = {
        "input_pad_tokens": _count(pad_positions),
        "attention_pad_tokens": _count(attention_mask.eq(0)),
        "ignored_labels": _count(ignored_labels),
        "supervised_labels": _count(supervised_labels),
        "shifted_supervised_labels": _count(shifted_supervised_labels),
        "supervised_pad_labels": _count(supervised_labels & labels.eq(pad_token_id)),
        "shifted_supervised_pad_labels": _count(shifted_supervised_labels & shifted_labels.eq(pad_token_id)),
        "pad_positions_not_ignored": _count(pad_positions & labels.ne(IGNORE_INDEX)),
        "masked_positions_not_ignored": _count(attention_mask.eq(0) & labels.ne(IGNORE_INDEX)),
    }

    pack_segment_ids = batch.get("pack_segment_ids")
    if isinstance(pack_segment_ids, torch.Tensor):
        stats["pack_padding_tokens"] = _count(pack_segment_ids.eq(0))
        stats["pack_padding_not_ignored"] = _count(pack_segment_ids.eq(0) & labels.ne(IGNORE_INDEX))

    print(f"batch {batch_index}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    failures = {
        key: value
        for key, value in stats.items()
        if key
        in {
            "supervised_pad_labels",
            "shifted_supervised_pad_labels",
            "pad_positions_not_ignored",
            "masked_positions_not_ignored",
            "pack_padding_not_ignored",
        }
        and value != 0
    }
    if failures:
        details = ", ".join(f"{key}={value}" for key, value in failures.items())
        raise RuntimeError(f"Batch {batch_index} has padding in supervised loss positions: {details}")

    return stats


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    import torch

    for parameter in model.parameters():
        if torch.is_floating_point(parameter):
            return parameter.dtype
    return torch.float32


def _autocast_dtype(config: PanguvlConfig) -> torch.dtype:
    import torch

    precision = str(config.optim.mixed_precision)
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _prepare_forward_batch(
    batch: dict[str, Any],
    *,
    config: PanguvlConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    import torch

    from recipes.panguvl.model.packing import prepare_packed_model_inputs

    forward_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            forward_batch[key] = value.to(device, non_blocking=True)
        else:
            forward_batch[key] = value

    forward_batch = prepare_packed_model_inputs(
        forward_batch,
        model_config=model.config,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        mask_dtype=_model_dtype(model),
    )

    dtype = _autocast_dtype(config)
    if dtype != torch.float32:
        for key in ("pixel_values", "pixel_values_videos"):
            value = forward_batch.get(key)
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                forward_batch[key] = value.to(dtype=dtype)
    return forward_batch


def _inspect_forward_loss(
    batch: dict[str, Any],
    *,
    config: PanguvlConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    batch_index: int,
) -> None:
    import torch

    forward_batch = _prepare_forward_batch(batch, config=config, model=model, device=device)
    autocast_dtype = _autocast_dtype(config)
    autocast_enabled = device.type != "cpu" and autocast_dtype != torch.float32

    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ),
    ):
        outputs = model(**forward_batch)

    loss = outputs.loss.detach()
    finite_mask = torch.isfinite(loss)
    nan_mask = torch.isnan(loss)
    inf_mask = torch.isinf(loss)
    print("  forward_loss:")
    print(f"    shape: {tuple(loss.shape)}")
    print(f"    finite: {_count(finite_mask)}")
    print(f"    nan: {_count(nan_mask)}")
    print(f"    inf: {_count(inf_mask)}")

    if bool(finite_mask.any().item()):
        finite_loss = loss[finite_mask].float()
        print(f"    finite_mean: {float(finite_loss.mean().item()):.6g}")
        print(f"    finite_max: {float(finite_loss.max().item()):.6g}")

    nonfinite_indices = torch.nonzero(~finite_mask, as_tuple=False).flatten()
    if nonfinite_indices.numel() == 0:
        return

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    shifted_labels = _shift_labels(batch["labels"])
    sequence_length = int(shifted_labels.shape[1])
    pack_segment_ids = batch.get("pack_segment_ids")

    print("    nonfinite_examples:")
    for flat_index in nonfinite_indices[:10].detach().cpu().tolist():
        sample_index = flat_index // sequence_length
        position = flat_index % sequence_length
        predicted_from_id = _as_int(input_ids[sample_index, position])
        shifted_label = _as_int(shifted_labels[sample_index, position])
        next_input_id = _as_int(input_ids[sample_index, min(position + 1, sequence_length - 1)])
        attention_value = _as_int(attention_mask[sample_index, position])
        segment_value = None
        if isinstance(pack_segment_ids, torch.Tensor):
            segment_value = _as_int(pack_segment_ids[sample_index, position])
        print(
            "      "
            f"flat={flat_index} sample={sample_index} pos={position} "
            f"loss={float(loss.flatten()[flat_index].float().item())} "
            f"input_id={predicted_from_id} next_input_id={next_input_id} "
            f"shifted_label={shifted_label} attention={attention_value} segment={segment_value}"
        )
        print(f"        context: {_decode_window(tokenizer, input_ids, sample_index, position)}")

    raise RuntimeError(f"Batch {batch_index} produced nonfinite per-token loss.")


def _init_diagnostic_logger() -> None:
    from mvp_engine.utils.log import init_logger
    from mvp_engine.utils.log.backend import TerminalBackend

    init_logger([TerminalBackend(id="panguvl-pad-loss-inspect")], interval=1, accumulation_size=1)


def _build_loader(config: PanguvlConfig, processor: Any, *, num_workers: int) -> Any:
    import torch
    from mvp_dataset import TorchLoader

    from recipes.panguvl.dataset import PanguvlCollator, build_dataset

    dataset = build_dataset(config, processor=processor)
    collate_fn = PanguvlCollator(
        pad_token_id=int(processor.tokenizer.pad_token_id),
        processor=processor,
    )
    loader = TorchLoader(
        dataset,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return loader.batch(
        batch_size=int(config.data.batch_size),
        drop_last=True,
        collate_fn=collate_fn,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="recipes/panguvl/configs/stage1.yaml", help="PanguVL YAML config.")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of real batches to inspect.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Device for optional forward pass.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for diagnostics. Defaults to 0 so failures surface in-process.",
    )
    parser.add_argument("--no-forward", action="store_true", help="Only inspect data/labels; do not load the model.")
    args = parser.parse_args()

    if args.num_batches < 1:
        raise ValueError("--num-batches must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    import torch

    from mvp_engine.launch import _apply_runtime_patches
    from recipes.panguvl.dataset import build_qwen3_vl_processor

    _apply_runtime_patches()
    _init_diagnostic_logger()

    config = _load_config(Path(args.config))
    processor = build_qwen3_vl_processor(config.model)
    pad_token_id = int(processor.tokenizer.pad_token_id)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    model = None
    if not args.no_forward:
        from recipes.panguvl.model import build_qwen3_vl_model

        model = build_qwen3_vl_model(config.model).to(requested_device)
        model.eval()

    _print_pad_configuration(config, processor, model)
    loader = _build_loader(config, processor, num_workers=int(args.num_workers))

    seen_batches = 0
    for batch_index, batch in enumerate(loader, start=1):
        _inspect_batch(batch, pad_token_id=pad_token_id, batch_index=batch_index)
        if model is not None:
            _inspect_forward_loss(
                batch,
                config=config,
                model=model,
                tokenizer=processor.tokenizer,
                device=requested_device,
                batch_index=batch_index,
            )
        seen_batches += 1
        if seen_batches >= args.num_batches:
            break

    if seen_batches != args.num_batches:
        raise RuntimeError(f"Only inspected {seen_batches} batches; requested {args.num_batches}.")

    print(f"inspection complete: {seen_batches} batch(es), no supervised pad-token labels found.")


if __name__ == "__main__":
    main()
