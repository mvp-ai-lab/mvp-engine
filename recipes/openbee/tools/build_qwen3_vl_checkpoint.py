"""Build the OpenBee Qwen3-VL-8B checkpoint.

This keeps the Qwen3-VL-8B-Instruct model structure and visual tower weights,
replaces the text backbone and LM head with Qwen3-8B-Base weights, and
re-initializes the multimodal merger.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

DEFAULT_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-8B-Base"
DEFAULT_OUTPUT_DIR = Path("recipes/openbee/pretrained/Qwen3-VL-8B-Instruct")
IGNORE_COPY_NAMES = {".cache", ".git", ".gitignore"}
MERGER_PREFIX = "model.visual.merger."
LANGUAGE_MODEL_PREFIX = "model.language_model."
LANGUAGE_MODEL_SOURCE_PREFIX = "model."
LM_HEAD_PREFIX = "lm_head."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vl-model-name-or-path", default=DEFAULT_VL_MODEL)
    parser.add_argument("--llm-model-name-or-path", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def resolve_model_path(name_or_path: str, cache_dir: Path | None, allow_patterns: list[str] | None = None) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path.resolve()

    snapshot_path = snapshot_download(
        repo_id=name_or_path,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=allow_patterns,
    )
    return Path(snapshot_path).resolve()


def copy_repo_tree(source_dir: Path, target_dir: Path) -> None:
    for source_path in sorted(source_dir.rglob("*")):
        relative_path = source_path.relative_to(source_dir)
        if any(part in IGNORE_COPY_NAMES for part in relative_path.parts):
            continue

        target_path = target_dir / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def load_weight_map(model_dir: Path) -> tuple[dict[str, str], str | None]:
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_files:
        if len(index_files) != 1:
            raise ValueError(f"Expected exactly one safetensors index file in {model_dir}, found {len(index_files)}.")

        with index_files[0].open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        return index["weight_map"], index_files[0].name

    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}.")

    weight_map: dict[str, str] = {}
    for shard_file in shard_files:
        with safe_open(shard_file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = shard_file.name
    return weight_map, None


def invert_weight_map(weight_map: dict[str, str]) -> dict[str, list[str]]:
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for tensor_name, file_name in weight_map.items():
        keys_by_file[file_name].append(tensor_name)

    for keys in keys_by_file.values():
        keys.sort()
    return dict(keys_by_file)


def classify_modified_key(tensor_name: str) -> str | None:
    if tensor_name.startswith(LANGUAGE_MODEL_PREFIX):
        return "language_model"
    if tensor_name.startswith(LM_HEAD_PREFIX):
        return "lm_head"
    if tensor_name.startswith(MERGER_PREFIX):
        return "merger"
    return None


def get_base_tensor_name(vl_tensor_name: str) -> str:
    if vl_tensor_name.startswith(LANGUAGE_MODEL_PREFIX):
        suffix = vl_tensor_name.removeprefix(LANGUAGE_MODEL_PREFIX)
        return f"{LANGUAGE_MODEL_SOURCE_PREFIX}{suffix}"

    if vl_tensor_name.startswith(LM_HEAD_PREFIX):
        return vl_tensor_name

    raise KeyError(f"Unsupported tensor remap request: {vl_tensor_name}")


def load_tensor_metadata(file_path: Path) -> dict[str, str]:
    with safe_open(file_path, framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def build_random_merger_tensor(
    tensor_name: str,
    reference_tensor: torch.Tensor,
    initializer_range: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if tensor_name.endswith(".norm.weight"):
        return torch.ones_like(reference_tensor)
    if tensor_name.endswith(".bias") or tensor_name.endswith(".norm.bias"):
        return torch.zeros_like(reference_tensor)
    if tensor_name.endswith(".weight"):
        random_tensor = torch.empty(reference_tensor.shape, dtype=torch.float32)
        random_tensor.normal_(mean=0.0, std=initializer_range, generator=generator)
        return random_tensor.to(dtype=reference_tensor.dtype)

    raise KeyError(f"Unsupported merger tensor for re-initialization: {tensor_name}")


def rewrite_output_shards(
    output_dir: Path,
    llm_source_dir: Path,
    vl_weight_map: dict[str, str],
    llm_weight_map: dict[str, str],
    seed: int,
) -> dict[str, Any]:
    vl_keys_by_file = invert_weight_map(vl_weight_map)
    llm_initializer_range = 0.02
    config_path = output_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        vl_config = json.load(handle)
    llm_initializer_range = float(vl_config["text_config"].get("initializer_range", 0.02))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    modified_stats = {
        "language_model_tensors": 0,
        "lm_head_tensors": 0,
        "merger_tensors": 0,
        "rewritten_shards": 0,
    }

    current_source_file_path: Path | None = None
    current_source_tensors: dict[str, torch.Tensor] | None = None
    for shard_name, shard_tensor_names in sorted(vl_keys_by_file.items()):
        modified_tensor_names = [name for name in shard_tensor_names if classify_modified_key(name) is not None]
        if not modified_tensor_names:
            continue

        output_shard_path = output_dir / shard_name
        shard_metadata = load_tensor_metadata(output_shard_path)
        target_tensors = load_file(str(output_shard_path))

        source_tensor_names_by_file: dict[Path, list[tuple[str, str]]] = defaultdict(list)
        for tensor_name in modified_tensor_names:
            tensor_kind = classify_modified_key(tensor_name)
            if tensor_kind == "merger":
                continue

            source_tensor_name = get_base_tensor_name(tensor_name)
            source_file_name = llm_weight_map.get(source_tensor_name)
            if source_file_name is None:
                raise KeyError(f"Missing source tensor {source_tensor_name} in LLM checkpoint.")
            source_file_path = llm_source_dir / source_file_name
            source_tensor_names_by_file[source_file_path].append((tensor_name, source_tensor_name))

        for source_file_path, tensor_name_pairs in sorted(source_tensor_names_by_file.items()):
            if current_source_file_path != source_file_path:
                current_source_tensors = load_file(str(source_file_path))
                current_source_file_path = source_file_path
            assert current_source_tensors is not None

            for output_tensor_name, source_tensor_name in tensor_name_pairs:
                source_tensor = current_source_tensors[source_tensor_name]
                target_tensor = target_tensors[output_tensor_name]
                if target_tensor.shape != source_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {output_tensor_name}: "
                        f"{tuple(target_tensor.shape)} != "
                        f"{tuple(source_tensor.shape)}"
                    )
                target_tensors[output_tensor_name] = source_tensor.to(dtype=target_tensor.dtype)
                tensor_kind = classify_modified_key(output_tensor_name)
                if tensor_kind == "language_model":
                    modified_stats["language_model_tensors"] += 1
                elif tensor_kind == "lm_head":
                    modified_stats["lm_head_tensors"] += 1

        for tensor_name in modified_tensor_names:
            if classify_modified_key(tensor_name) != "merger":
                continue

            target_tensors[tensor_name] = build_random_merger_tensor(
                tensor_name=tensor_name,
                reference_tensor=target_tensors[tensor_name],
                initializer_range=llm_initializer_range,
                generator=generator,
            )
            modified_stats["merger_tensors"] += 1

        save_file(target_tensors, str(output_shard_path), metadata=shard_metadata)
        modified_stats["rewritten_shards"] += 1

    return modified_stats


def write_build_metadata(
    output_dir: Path,
    vl_model_name_or_path: str,
    llm_model_name_or_path: str,
    seed: int,
    stats: dict[str, Any],
) -> None:
    metadata = {
        "vl_model_name_or_path": vl_model_name_or_path,
        "llm_model_name_or_path": llm_model_name_or_path,
        "seed": seed,
        "strategy": {
            "keep_visual_tower": True,
            "replace_language_model": True,
            "replace_lm_head": True,
            "random_init_merger": True,
        },
        "stats": stats,
    }

    metadata_path = output_dir / "openbee_checkpoint_build.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists. Pass --force to overwrite it.")
        shutil.rmtree(output_dir)

    vl_source_dir = resolve_model_path(args.vl_model_name_or_path, args.cache_dir)
    llm_source_dir = resolve_model_path(
        args.llm_model_name_or_path,
        args.cache_dir,
        allow_patterns=["*.json", "*.safetensors", "*.safetensors.index.json"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_repo_tree(vl_source_dir, output_dir)

    vl_weight_map, _ = load_weight_map(output_dir)
    llm_weight_map, _ = load_weight_map(llm_source_dir)

    stats = rewrite_output_shards(
        output_dir=output_dir,
        llm_source_dir=llm_source_dir,
        vl_weight_map=vl_weight_map,
        llm_weight_map=llm_weight_map,
        seed=args.seed,
    )
    write_build_metadata(
        output_dir=output_dir,
        vl_model_name_or_path=args.vl_model_name_or_path,
        llm_model_name_or_path=args.llm_model_name_or_path,
        seed=args.seed,
        stats=stats,
    )

    print(json.dumps({"output_dir": str(output_dir), **stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
