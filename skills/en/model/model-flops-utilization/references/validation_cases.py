"""Validation cases for model-flops-utilization skill routing and contract checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ValidationCase:
    prompt_id: str
    model_name: str
    case_type: str  # implementation | boundary
    prompt: str
    expected_trigger: str


CASES: list[ValidationCase] = [
    ValidationCase(
        prompt_id="T1",
        model_name="bert-base-uncased",
        case_type="implementation",
        prompt="Please add calculate_model_flops(...) to this encoder Transformer model and return FLOPs per training step as a float.",
        expected_trigger="model-flops-utilization",
    ),
    ValidationCase(
        prompt_id="T2",
        model_name="bert-base-uncased",
        case_type="boundary",
        prompt="Implement calculate_model_flops for Transformer but do not require seq_len.",
        expected_trigger="correct_to_skill_contract",
    ),
    ValidationCase(
        prompt_id="T3",
        model_name="gpt2",
        case_type="implementation",
        prompt="Refactor this decoder-only model so calculate_model_flops handles is_training=True/False and documents lm_head inclusion assumptions.",
        expected_trigger="model-flops-utilization",
    ),
    ValidationCase(
        prompt_id="T4",
        model_name="gpt2",
        case_type="boundary",
        prompt="Return a breakdown dict only; no float needed.",
        expected_trigger="correct_to_skill_contract",
    ),
    ValidationCase(
        prompt_id="T5",
        model_name="t5-small",
        case_type="implementation",
        prompt="Implement MFU support for this encoder-decoder model by adding calculate_model_flops with explicit batch_size and seq_len inputs.",
        expected_trigger="model-flops-utilization",
    ),
    ValidationCase(
        prompt_id="T6",
        model_name="t5-small",
        case_type="boundary",
        prompt="For this model, avoid explicit shape parameters and infer everything from runtime state.",
        expected_trigger="correct_to_skill_contract",
    ),
    ValidationCase(
        prompt_id="T7",
        model_name="google/vit-base-patch16-224",
        case_type="implementation",
        prompt="Add calculate_model_flops(...) for this ViT classifier using batch_size, image_size, patch_size, and is_training.",
        expected_trigger="model-flops-utilization",
    ),
    ValidationCase(
        prompt_id="T8",
        model_name="google/vit-base-patch16-224",
        case_type="boundary",
        prompt="For ViT, infer patch_size from hidden states at runtime and avoid method parameters.",
        expected_trigger="correct_to_skill_contract",
    ),
    ValidationCase(
        prompt_id="T9",
        model_name="facebook/deit-tiny-patch16-224",
        case_type="implementation",
        prompt="Implement calculate_model_flops(...) for this external tiny ViT model with explicit batch_size, image_size, patch_size, and is_training.",
        expected_trigger="model-flops-utilization",
    ),
    ValidationCase(
        prompt_id="T10",
        model_name="facebook/deit-tiny-patch16-224",
        case_type="boundary",
        prompt="For this external tiny ViT model, return dict-only output and skip explicit shape parameters.",
        expected_trigger="correct_to_skill_contract",
    ),
]


def _render_list() -> None:
    for case in CASES:
        print(f"{case.prompt_id}\t{case.case_type}\t{case.model_name}\t{case.expected_trigger}\t{case.prompt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="List validation cases for MFU skill checks.")
    parser.add_argument("--list", action="store_true", help="Print tab-separated validation cases.")
    parser.add_argument("--json", action="store_true", help="Print validation cases as JSON.")
    args = parser.parse_args()

    if args.json:
        print(json.dumps([asdict(c) for c in CASES], ensure_ascii=False, indent=2))
        return 0

    _render_list()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
