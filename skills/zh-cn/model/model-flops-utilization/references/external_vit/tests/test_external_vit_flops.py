from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

from transformers import ViTConfig


def _load_model_class():
    model_file = Path(__file__).resolve().parents[1] / "modeling_vit_with_flops.py"
    spec = importlib.util.spec_from_file_location("external_vit_modeling_zh", model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ViTForImageClassificationWithFlops


def main() -> int:
    model_cls = _load_model_class()
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=192,
        intermediate_size=768,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_labels=1000,
    )
    model = model_cls(config)

    fn = model.calculate_model_flops
    sig = inspect.signature(fn)
    required_names = {"batch_size", "image_size", "patch_size", "is_training"}
    missing = [name for name in required_names if name not in sig.parameters]
    if missing:
        raise SystemExit(f"[FAIL] signature missing required parameters: {missing}")

    train_flops = fn(batch_size=2, image_size=224, patch_size=16, is_training=True)
    eval_flops = fn(batch_size=2, image_size=224, patch_size=16, is_training=False)

    if not isinstance(train_flops, float) or not isinstance(eval_flops, float):
        raise SystemExit("[FAIL] calculate_model_flops must return float for both train and eval")
    if train_flops <= 0 or eval_flops <= 0:
        raise SystemExit("[FAIL] FLOPs must be positive")
    if train_flops < eval_flops:
        raise SystemExit("[FAIL] training FLOPs should be >= eval FLOPs")

    try:
        fn(batch_size=2, patch_size=16, is_training=True)
    except (ValueError, TypeError):
        pass
    else:
        raise SystemExit("[FAIL] missing image_size should raise ValueError/TypeError")

    print("[PASS] external ViT 参考合同测试通过")
    print(f"train_flops={train_flops:.3e}")
    print(f"eval_flops={eval_flops:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
