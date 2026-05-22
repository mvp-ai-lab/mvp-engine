"""Validate that a PanguVL checkpoint loads without newly initialized weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KEY_PREFIXES = (
    "model.layers.",
    "model.language_model.layers.",
    "model.embed_tokens.",
    "model.language_model.embed_tokens.",
    "lm_head.",
    "visual.",
    "model.visual.",
    "model.visual.merger.",
    "model.visual.vision_projection.",
)

OPENPANGU_KEY_MAPPING = {
    "^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}

ALLOW_REINITIALIZED_PREFIXES = (
    "model.visual.merger.",
    "model.visual.vision_projection.",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint_keys(checkpoint_dir: Path) -> set[str]:
    index_files = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
    if index_files:
        if len(index_files) != 1:
            raise ValueError(f"Expected exactly one safetensors index in {checkpoint_dir}, found {len(index_files)}.")
        index = load_json(index_files[0])
        return set(index["weight_map"])

    shard_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors index or shard files found in {checkpoint_dir}.")

    from safetensors import safe_open

    keys: set[str] = set()
    for shard_file in shard_files:
        with safe_open(shard_file, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def count_prefixes(keys: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for key in keys:
        matched = False
        for prefix in KEY_PREFIXES:
            if key.startswith(prefix):
                counts[prefix] += 1
                matched = True
        if not matched:
            counts["<other>"] += 1
    return counts


def remap_openpangu_key(key: str) -> str:
    """Apply the OpenPangu remote-code checkpoint mapping to one tensor key."""
    if key.startswith("visual"):
        return f"model.{key}"
    if key.startswith("model.") and not key.startswith(("model.language_model.", "model.visual.")):
        return f"model.language_model{key.removeprefix('model')}"
    return key


def remap_checkpoint_keys(keys: set[str]) -> dict[str, str]:
    remapped: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for key in sorted(keys):
        target = remap_openpangu_key(key)
        previous = remapped.get(target)
        if previous is not None:
            collisions.setdefault(target, [previous]).append(key)
            continue
        remapped[target] = key
    if collisions:
        formatted = ", ".join(f"{target} <- {sources}" for target, sources in sorted(collisions.items())[:20])
        raise RuntimeError(f"OpenPangu remap has tensor-key collisions: {formatted}")
    return remapped


def requires_openpangu_remap(keys: set[str]) -> bool:
    return any(key.startswith(("model.layers.", "model.embed_tokens.")) for key in keys) and not any(
        key.startswith(("model.language_model.layers.", "model.language_model.embed_tokens.")) for key in keys
    )


def summarize_counts(title: str, keys: set[str]) -> None:
    print(f"{title}:")
    counts = count_prefixes(keys)
    for prefix in KEY_PREFIXES:
        print(f"  {prefix}: {counts[prefix]}")
    print(f"  <other>: {counts['<other>']}")
    print(f"  total: {len(keys)}")


def normalize_loading_info(loading_info: dict[str, Any]) -> dict[str, set[str]]:
    normalized: dict[str, set[str]] = {}
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        values = loading_info.get(name, set())
        normalized[name] = {str(value[0] if isinstance(value, (tuple, list)) and value else value) for value in values}
    conversion_errors = loading_info.get("conversion_errors", {})
    if isinstance(conversion_errors, dict):
        normalized["conversion_errors"] = {str(key) for key in conversion_errors}
    else:
        normalized["conversion_errors"] = {str(value) for value in conversion_errors}
    return normalized


def is_allowed_reinitialized_key(key: str, *, allow_merger_init: bool) -> bool:
    if allow_merger_init and key.startswith(ALLOW_REINITIALIZED_PREFIXES):
        return True
    return False


def classify_failures(
    loading_info: dict[str, set[str]],
    *,
    allow_merger_init: bool,
) -> dict[str, set[str]]:
    missing = {
        key
        for key in loading_info["missing_keys"]
        if not is_allowed_reinitialized_key(key, allow_merger_init=allow_merger_init)
    }
    unexpected = set(loading_info["unexpected_keys"])
    mismatched = set(loading_info["mismatched_keys"])
    conversion_errors = set(loading_info["conversion_errors"])

    hard_missing = {
        key
        for key in missing
        if key.startswith(("model.language_model.", "model.visual.", "lm_head.")) or key == "lm_head.weight"
    }
    hard_unexpected = {key for key in unexpected if key.startswith(("model.", "visual.", "lm_head."))}

    return {
        "missing_keys": hard_missing,
        "unexpected_keys": hard_unexpected,
        "mismatched_keys": mismatched,
        "conversion_errors": conversion_errors,
    }


def print_key_examples(title: str, keys: set[str], *, limit: int = 20) -> None:
    if not keys:
        return
    print(f"{title} ({len(keys)}):")
    for key in sorted(keys)[:limit]:
        print(f"  {key}")
    if len(keys) > limit:
        print(f"  ... {len(keys) - limit} more")


def validate_metadata(checkpoint_dir: Path, checkpoint_keys: set[str]) -> None:
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}.")

    config = load_json(config_path)
    print("checkpoint metadata:")
    print(f"  path: {checkpoint_dir}")
    print(f"  model_type: {config.get('model_type')}")
    print(f"  architectures: {config.get('architectures')}")
    print(f"  auto_map.AutoModelForCausalLM: {(config.get('auto_map') or {}).get('AutoModelForCausalLM')}")

    summarize_counts("checkpoint keyspace", checkpoint_keys)
    print(f"requires OpenPangu namespace remap: {requires_openpangu_remap(checkpoint_keys)}")

    remapped = remap_checkpoint_keys(checkpoint_keys)
    summarize_counts("checkpoint keyspace after simulated OpenPangu remap", set(remapped))

    if "lm_head.weight" not in checkpoint_keys:
        raise RuntimeError("Checkpoint is missing lm_head.weight.")


def load_model_for_validation(
    checkpoint_dir: Path,
    *,
    attn_implementation: str | None,
    device: str,
    use_key_mapping: bool,
) -> tuple[Any, dict[str, set[str]]]:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "output_loading_info": True,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if device:
        kwargs["device_map"] = {"": device}
    if use_key_mapping:
        kwargs["key_mapping"] = OPENPANGU_KEY_MAPPING

    model, loading_info = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir), **kwargs)
    return model, normalize_loading_info(loading_info)


def validate_model_keyspace(checkpoint_keys: set[str], model_keys: set[str]) -> None:
    remapped = remap_checkpoint_keys(checkpoint_keys)
    remapped_keys = set(remapped)
    missing_after_remap = remapped_keys - model_keys
    extra_model_pretrained_keys = {
        key
        for key in model_keys - remapped_keys
        if key.startswith(("model.language_model.", "model.visual.", "lm_head."))
    }

    summarize_counts("instantiated model keyspace", model_keys)
    print_key_examples("remapped checkpoint keys absent from model", missing_after_remap)
    print_key_examples("model pretrained-looking keys absent from remapped checkpoint", extra_model_pretrained_keys)

    if missing_after_remap:
        raise RuntimeError("Simulated OpenPangu key remap produced keys that do not exist in the instantiated model.")


def validate_loading_info(loading_info: dict[str, set[str]], *, allow_merger_init: bool) -> None:
    print("loading info:")
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "conversion_errors"):
        print(f"  {name}: {len(loading_info[name])}")
        print_key_examples(f"  {name} examples", loading_info[name], limit=10)

    failures = classify_failures(loading_info, allow_merger_init=allow_merger_init)
    failing_categories = {name: keys for name, keys in failures.items() if keys}
    if failing_categories:
        for name, keys in failing_categories.items():
            print_key_examples(f"disallowed {name}", keys)
        raise RuntimeError("Checkpoint load has disallowed missing, unexpected, mismatched, or conversion-error keys.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Local PanguVL/OpenPangu checkpoint dir.")
    parser.add_argument("--attn-implementation", default=None, help="Optional attention implementation for model load.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Device map for full validation load.")
    parser.add_argument(
        "--metadata-only", action="store_true", help="Only inspect config/index metadata; do not load model."
    )
    parser.add_argument(
        "--use-key-mapping", action="store_true", help="Pass explicit OpenPangu key_mapping to from_pretrained."
    )
    parser.add_argument(
        "--allow-merger-init",
        action="store_true",
        help="Allow merger/projection tensors to be newly initialized for intentional hybrid checkpoints.",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_keys = load_checkpoint_keys(checkpoint_dir)
    validate_metadata(checkpoint_dir, checkpoint_keys)

    if args.metadata_only:
        print("metadata validation complete.")
        return

    model, loading_info = load_model_for_validation(
        checkpoint_dir,
        attn_implementation=args.attn_implementation,
        device=args.device,
        use_key_mapping=bool(args.use_key_mapping),
    )
    model_keys = set(model.state_dict())
    validate_model_keyspace(checkpoint_keys, model_keys)
    validate_loading_info(loading_info, allow_merger_init=bool(args.allow_merger_init))
    print("checkpoint load validation complete: no disallowed newly initialized weights detected.")


if __name__ == "__main__":
    main()
