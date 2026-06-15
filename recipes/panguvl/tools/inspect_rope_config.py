"""Inspect whether OpenPangu mRoPE config survives model loading."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_MROPE_SECTION = [10, 27, 27]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_recipe_config(config_path: Path) -> Any:
    from omegaconf import OmegaConf

    from recipes.panguvl.configs.schema import PanguvlConfig

    raw_config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return PanguvlConfig.model_validate(raw_config)


def checkpoint_dir_from_args(args: argparse.Namespace, config: Any) -> Path:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir.expanduser().resolve()
    return Path(config.model.pretrained_model_name_or_path).expanduser().resolve()


def assert_mrope_scaling(rope_scaling: dict[str, Any] | None, *, source: str) -> None:
    if rope_scaling is None:
        raise RuntimeError(f"{source} has no rope_scaling/rope_parameters.")

    section = rope_scaling.get("mrope_section")
    interleaved = rope_scaling.get("mrope_interleaved")
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if section != EXPECTED_MROPE_SECTION:
        raise RuntimeError(f"{source} mrope_section mismatch: expected {EXPECTED_MROPE_SECTION}, got {section}.")
    if interleaved is not True:
        raise RuntimeError(f"{source} mrope_interleaved mismatch: expected True, got {interleaved}.")
    if rope_type != "default":
        raise RuntimeError(f"{source} rope_type mismatch: expected 'default', got {rope_type!r}.")


def print_rope_scaling(title: str, rope_scaling: dict[str, Any] | None) -> None:
    print(f"{title}:")
    if rope_scaling is None:
        print("  <missing>")
        return
    for key in ("rope_type", "type", "mrope_section", "mrope_interleaved", "rope_theta"):
        if key in rope_scaling:
            print(f"  {key}: {rope_scaling[key]}")


def get_text_rope_scaling(config: Any) -> dict[str, Any] | None:
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return getattr(config, "rope_scaling", None)
    return getattr(text_config, "rope_scaling", None)


def get_rotary_embedding(model: Any) -> Any:
    return model.model.language_model.rotary_emb


def print_rotary_fields(rotary_emb: Any) -> None:
    print("rotary embedding:")
    print(f"  class: {rotary_emb.__class__.__module__}.{rotary_emb.__class__.__qualname__}")
    print(f"  rope_type: {getattr(rotary_emb, 'rope_type', None)}")
    print(f"  mrope_section: {getattr(rotary_emb, 'mrope_section', None)}")
    print(f"  mrope_interleaved: {getattr(rotary_emb, 'mrope_interleaved', None)}")
    print(f"  has_mrope_dim: {hasattr(rotary_emb, 'mrope_dim')}")
    if hasattr(rotary_emb, "mrope_dim"):
        print(f"  mrope_dim_len: {len(rotary_emb.mrope_dim)}")


def _source_path(obj: Any) -> str:
    try:
        source = inspect.getsourcefile(obj) or inspect.getfile(obj)
    except TypeError:
        return "<unknown>"
    return source or "<unknown>"


def _source_text(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""


def print_rotary_provenance(rotary_emb: Any) -> None:
    cls = rotary_emb.__class__
    class_source = _source_path(cls)
    forward_source = _source_text(cls.forward)
    init_source = _source_text(cls.__init__)
    rope_init_fn = getattr(rotary_emb, "rope_init_fn", None)

    print("rotary provenance:")
    print(f"  class_name: {cls.__name__}")
    print(f"  class_module: {cls.__module__}")
    print(f"  class_source: {class_source}")
    print(f"  is_openpangu_rotary_class: {cls.__name__ == 'OpenPanguVLRotaryEmbedding'}")
    print(f"  source_is_openpangu_modeling: {class_source.endswith('modeling_openpangu_vl.py')}")
    print(f"  init_reads_mrope_section: {'mrope_section' in init_source}")
    print(f"  init_reads_mrope_interleaved: {'mrope_interleaved' in init_source}")
    print(f"  forward_uses_mrope_interleaved: {'mrope_interleaved' in forward_source}")
    print(f"  forward_uses_mrope_dim: {'mrope_dim' in forward_source}")
    if rope_init_fn is not None:
        print(f"  rope_init_fn: {rope_init_fn.__module__}.{rope_init_fn.__name__}")
        print(f"  rope_init_fn_source: {_source_path(rope_init_fn)}")


def assert_openpangu_rotary_source(rotary_emb: Any) -> None:
    cls = rotary_emb.__class__
    class_source = _source_path(cls)
    if cls.__name__ != "OpenPanguVLRotaryEmbedding":
        raise RuntimeError(f"Expected OpenPanguVLRotaryEmbedding, got {cls.__module__}.{cls.__name__}.")
    if not class_source.endswith("modeling_openpangu_vl.py"):
        raise RuntimeError(f"Expected OpenPangu modeling_openpangu_vl.py rotary source, got {class_source}.")
    forward_source = _source_text(cls.forward)
    if "mrope_interleaved" not in forward_source or "mrope_dim" not in forward_source:
        raise RuntimeError("OpenPangu rotary forward does not appear to use mRoPE routing fields.")


def assert_rotary_mrope_active(rotary_emb: Any) -> None:
    section = getattr(rotary_emb, "mrope_section", None)
    interleaved = getattr(rotary_emb, "mrope_interleaved", None)
    if section != EXPECTED_MROPE_SECTION:
        raise RuntimeError(f"rotary_emb.mrope_section mismatch: expected {EXPECTED_MROPE_SECTION}, got {section}.")
    if interleaved is not True:
        raise RuntimeError(f"rotary_emb.mrope_interleaved mismatch: expected True, got {interleaved}.")
    if not hasattr(rotary_emb, "mrope_dim"):
        raise RuntimeError("rotary_emb.mrope_dim is missing even though interleaved mRoPE is enabled.")


def build_model(config: Any, checkpoint_dir: Path, *, device: str) -> Any:
    from transformers import AutoModelForCausalLM

    from recipes.panguvl.model.qwen3_vl import OPENPANGU_KEY_MAPPING

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation=config.model.attn_implementation,
        key_mapping=OPENPANGU_KEY_MAPPING,
        output_loading_info=True,
        device_map={"": device},
    )
    if loading_info.get("missing_keys") or loading_info.get("unexpected_keys") or loading_info.get("mismatched_keys"):
        print("loading info:")
        print(f"  missing_keys: {len(loading_info.get('missing_keys') or [])}")
        print(f"  unexpected_keys: {len(loading_info.get('unexpected_keys') or [])}")
        print(f"  mismatched_keys: {len(loading_info.get('mismatched_keys') or [])}")
    model.eval()
    return model


def apply_shape_preserving_mrope_control(model: Any) -> None:
    """Replace OpenPangu's interleaved mRoPE routing with a valid control routing.

    Removing mRoPE entirely leaves the rotary embedding with three position
    streams and expands attention dimensions. This control keeps the output
    shape valid while proving the configured interleaving affects logits.
    """
    rotary_emb = get_rotary_embedding(model)
    rotary_emb.mrope_interleaved = True
    rotary_emb.mrope_section = EXPECTED_MROPE_SECTION
    rotary_emb.mrope_dim = [0] * len(rotary_emb.mrope_dim)


def build_synthetic_inputs(model: Any, *, device: str, seq_len: int) -> dict[str, Any]:
    import torch

    config = model.config
    vocab_size = int(config.text_config.vocab_size)
    image_token_id = int(getattr(config, "image_token_id"))
    vision_start_token_id = int(getattr(config, "vision_start_token_id"))
    vision_end_token_id = int(getattr(config, "vision_end_token_id"))
    bos_token_id = int(getattr(config, "bos_token_id", 1))
    eos_token_id = int(getattr(config, "eos_token_id", 2))

    spatial_merge_size = int(config.vision_config.spatial_merge_size)
    image_grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    image_token_count = int(image_grid_thw.prod().item()) // (spatial_merge_size**2)
    image_start = 2
    image_end = image_start + image_token_count
    minimum_seq_len = image_end + 3
    if seq_len < minimum_seq_len:
        raise ValueError(f"--seq-len must be at least {minimum_seq_len} for the synthetic image probe.")

    input_ids = torch.arange(seq_len, device=device, dtype=torch.long).remainder(max(vocab_size - 10, 1)) + 10
    input_ids[0] = bos_token_id
    input_ids[1] = vision_start_token_id
    input_ids[image_start:image_end] = image_token_id
    input_ids[image_end] = vision_end_token_id
    input_ids[-1] = eos_token_id
    input_ids = input_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    patch_size = int(config.vision_config.patch_size)
    channels = int(getattr(config.vision_config, "in_chans", 3))
    pixel_dim = channels * patch_size * patch_size
    pixel_values = torch.linspace(
        0.0,
        1.0,
        steps=int(image_grid_thw.prod().item()) * pixel_dim,
        device=device,
        dtype=next(model.parameters()).dtype,
    ).reshape(int(image_grid_thw.prod().item()), pixel_dim)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def get_synthetic_position_ids(model: Any, inputs: dict[str, Any]) -> Any:
    position_ids, _ = model.model.get_rope_index(
        input_ids=inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs.get("attention_mask"),
    )
    return position_ids


def print_position_id_summary(position_ids: Any) -> None:
    import torch

    stream_gap = (position_ids.max(dim=0).values - position_ids.min(dim=0).values).max()
    print("synthetic position_ids:")
    print(f"  shape: {tuple(position_ids.shape)}")
    print(f"  min: {int(position_ids.min().item())}")
    print(f"  max: {int(position_ids.max().item())}")
    print(
        f"  streams_identical: {bool(torch.equal(position_ids[0], position_ids[1]) and torch.equal(position_ids[1], position_ids[2]))}"
    )
    print(f"  max_stream_gap: {int(stream_gap.item())}")


def run_forward(model: Any, inputs: dict[str, Any]) -> Any:
    import torch

    with torch.no_grad():
        return model(**inputs)


def benchmark_forward(model: Any, inputs: dict[str, Any], *, passes: int, device: str) -> dict[str, float]:
    import torch

    if passes <= 0:
        return {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(passes):
        output = run_forward(model, inputs)
        logits = output.logits if output.logits is not None else None
        if logits is not None:
            _ = float(logits[..., -1, :].float().mean().detach().cpu().item())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = int(inputs["input_ids"].numel()) * passes
    result = {
        "passes": float(passes),
        "elapsed_seconds": elapsed,
        "seconds_per_pass": elapsed / passes,
        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
    }
    if device == "cuda":
        result["max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return result


def print_benchmark(title: str, result: dict[str, float]) -> None:
    if not result:
        return
    print(title + ":")
    for key, value in result.items():
        if key == "passes":
            print(f"  {key}: {int(value)}")
        else:
            print(f"  {key}: {value:.6g}")


def compare_enabled_disabled(model: Any, inputs: dict[str, Any]) -> None:
    import torch

    rotary_emb = get_rotary_embedding(model)
    original_mrope_interleaved = getattr(rotary_emb, "mrope_interleaved", None)
    original_mrope_section = getattr(rotary_emb, "mrope_section", None)
    original_mrope_dim = getattr(rotary_emb, "mrope_dim", None)
    position_ids = get_synthetic_position_ids(model, inputs)
    hidden_probe = torch.empty(
        inputs["input_ids"].shape[0],
        inputs["input_ids"].shape[1],
        int(model.config.text_config.hidden_size),
        device=inputs["input_ids"].device,
        dtype=next(model.parameters()).dtype,
    )

    enabled_output = run_forward(model, inputs)
    enabled_logits = enabled_output.logits.detach().float()
    enabled_cos, enabled_sin = rotary_emb(hidden_probe, position_ids)
    enabled_cos = enabled_cos.detach().float()
    enabled_sin = enabled_sin.detach().float()

    try:
        apply_shape_preserving_mrope_control(model)
        control_cos, control_sin = rotary_emb(hidden_probe, position_ids)
        control_cos = control_cos.detach().float()
        control_sin = control_sin.detach().float()
        disabled_output = run_forward(model, inputs)
        disabled_logits = disabled_output.logits.detach().float()
    finally:
        rotary_emb.mrope_interleaved = original_mrope_interleaved
        rotary_emb.mrope_section = original_mrope_section
        if original_mrope_dim is not None:
            rotary_emb.mrope_dim = original_mrope_dim

    diff = (enabled_logits - disabled_logits).abs()
    cos_diff = (enabled_cos - control_cos).abs()
    sin_diff = (enabled_sin - control_sin).abs()
    print("enabled_vs_control_mrope:")
    print("  control: interleaved routing replaced with all-temporal routing")
    print(f"  max_abs_cos_diff: {float(cos_diff.max().item()):.6g}")
    print(f"  mean_abs_cos_diff: {float(cos_diff.mean().item()):.6g}")
    print(f"  max_abs_sin_diff: {float(sin_diff.max().item()):.6g}")
    print(f"  mean_abs_sin_diff: {float(sin_diff.mean().item()):.6g}")
    print(f"  max_abs_logit_diff: {float(diff.max().item()):.6g}")
    print(f"  mean_abs_logit_diff: {float(diff.mean().item()):.6g}")
    if not bool(torch.isfinite(diff).all().item()):
        raise RuntimeError("Enabled/control mRoPE comparison produced nonfinite differences.")
    if not bool(torch.isfinite(cos_diff).all().item()) or not bool(torch.isfinite(sin_diff).all().item()):
        raise RuntimeError("Enabled/control mRoPE comparison produced nonfinite rotary differences.")
    if float(cos_diff.max().item()) == 0.0 and float(sin_diff.max().item()) == 0.0:
        print("  control_result: inconclusive_zero_rotary_diff")
        return
    print("  control_result: changed_rotary_embeddings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="recipes/panguvl/configs/stage1.yaml", help="PanguVL YAML config.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Override checkpoint directory.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Device for optional model checks.")
    parser.add_argument("--load-model", action="store_true", help="Load model and inspect runtime rotary embedding.")
    parser.add_argument("--num-forward-passes", type=int, default=0, help="Benchmark no-grad forward passes.")
    parser.add_argument(
        "--seq-len", type=int, default=32, help="Synthetic sequence length for optional forward checks."
    )
    parser.add_argument(
        "--compare-disabled",
        action="store_true",
        help="Compare logits with a shape-preserving altered mRoPE routing control.",
    )
    args = parser.parse_args()

    if args.num_forward_passes < 0:
        raise ValueError("--num-forward-passes must be non-negative.")

    config = load_recipe_config(Path(args.config))
    checkpoint_dir = checkpoint_dir_from_args(args, config)
    checkpoint_config = load_json(checkpoint_dir / "config.json")
    raw_rope_scaling = checkpoint_config.get("rope_scaling")
    print_rope_scaling("raw checkpoint rope_scaling", raw_rope_scaling)
    assert_mrope_scaling(raw_rope_scaling, source="raw checkpoint config")
    print("raw checkpoint mRoPE fields: preserved")

    if not args.load_model and args.num_forward_passes == 0 and not args.compare_disabled:
        print("metadata-only rope inspection complete.")
        return

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    model = build_model(config, checkpoint_dir, device=args.device)
    loaded_rope_scaling = get_text_rope_scaling(model.config)
    print_rope_scaling("loaded text_config rope_scaling", loaded_rope_scaling)
    assert_mrope_scaling(loaded_rope_scaling, source="loaded text config")

    rotary_emb = get_rotary_embedding(model)
    print_rotary_fields(rotary_emb)
    print_rotary_provenance(rotary_emb)
    assert_openpangu_rotary_source(rotary_emb)
    assert_rotary_mrope_active(rotary_emb)
    print("runtime mRoPE fields: active")

    inputs = None
    if args.num_forward_passes > 0 or args.compare_disabled:
        inputs = build_synthetic_inputs(model, device=args.device, seq_len=int(args.seq_len))
        position_ids = get_synthetic_position_ids(model, inputs)
        print_position_id_summary(position_ids)
        output = run_forward(model, inputs)
        logits = output.logits
        if logits is None:
            raise RuntimeError("Forward output did not include logits.")
        finite = bool(torch.isfinite(logits).all().item())
        print("synthetic forward:")
        print(f"  logits_shape: {tuple(logits.shape)}")
        print(f"  logits_finite: {finite}")
        if not finite:
            raise RuntimeError("Synthetic forward produced nonfinite logits.")

    if args.compare_disabled:
        assert inputs is not None
        compare_enabled_disabled(model, inputs)

    if args.num_forward_passes > 0:
        assert inputs is not None
        benchmark = benchmark_forward(model, inputs, passes=int(args.num_forward_passes), device=args.device)
        print_benchmark("forward benchmark", benchmark)
    print("rope inspection complete.")


if __name__ == "__main__":
    main()
