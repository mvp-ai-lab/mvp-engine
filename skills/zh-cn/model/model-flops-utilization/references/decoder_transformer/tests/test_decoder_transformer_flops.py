from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

from transformers import GPT2Config


def _load_model_class():
    model_file = Path(__file__).resolve().parents[1] / "modeling_decoder_with_flops.py"
    spec = importlib.util.spec_from_file_location("decoder_transformer_modeling_zh", model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GPT2LMHeadModelWithFlops


def main() -> int:
    model_cls = _load_model_class()
    config = GPT2Config(n_layer=2, n_embd=128, n_head=4, vocab_size=4096)
    model = model_cls(config)

    fn = model.calculate_model_flops
    sig = inspect.signature(fn)
    required_names = {"batch_size", "seq_len", "is_training"}
    missing = [name for name in required_names if name not in sig.parameters]
    if missing:
        raise SystemExit(f"[FAIL] signature missing required parameters: {missing}")

    train_flops = fn(batch_size=2, seq_len=128, is_training=True)
    eval_flops = fn(batch_size=2, seq_len=128, is_training=False)

    if not isinstance(train_flops, float) or not isinstance(eval_flops, float):
        raise SystemExit("[FAIL] calculate_model_flops must return float for both train and eval")
    if train_flops <= 0 or eval_flops <= 0:
        raise SystemExit("[FAIL] FLOPs must be positive")
    if train_flops < eval_flops:
        raise SystemExit("[FAIL] training FLOPs should be >= eval FLOPs")

    try:
        fn(batch_size=2, is_training=True)
    except (ValueError, TypeError):
        pass
    else:
        raise SystemExit("[FAIL] missing seq_len should raise ValueError/TypeError")

    print("[PASS] decoder transformer 参考合同测试通过")
    print(f"train_flops={train_flops:.3e}")
    print(f"eval_flops={eval_flops:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
