import importlib
import sys
import types
from types import SimpleNamespace

import torch


def _install_test_stubs() -> None:
    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForImageTextToText = object
        sys.modules["transformers"] = transformers

    if "transformers.utils" not in sys.modules:
        sys.modules["transformers.utils"] = types.ModuleType("transformers.utils")

    if "transformers.utils.logging" not in sys.modules:
        logging_mod = types.ModuleType("transformers.utils.logging")
        logging_mod.disable_progress_bar = lambda: None
        sys.modules["transformers.utils.logging"] = logging_mod


_install_test_stubs()
openbee_mfu = importlib.import_module("recipes.openbee.utils.log.mfu")
qwen3_vl = importlib.import_module("recipes.openbee.model.qwen3_vl")

build_mfu_log = openbee_mfu.build_mfu_log
calculate_mfu = openbee_mfu.calculate_mfu
inject_model_flops_calculation = qwen3_vl.inject_model_flops_calculation
normalize_device_name = openbee_mfu.normalize_device_name


def test_inject_model_flops_calculation_supports_multimodal_inputs():
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                num_hidden_layers=2,
                hidden_size=16,
                intermediate_size=32,
                vocab_size=64,
            ),
            vision_config=SimpleNamespace(
                depth=3,
                hidden_size=8,
                intermediate_size=16,
                out_hidden_size=16,
                in_channels=3,
                patch_size=2,
                temporal_patch_size=1,
                spatial_merge_size=2,
            ),
        )
    )

    inject_model_flops_calculation(model)

    flops = model.calculate_model_flops(
        batch_size=2,
        seq_len=12,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 6, 6]]),
        is_training=True,
    )

    assert isinstance(flops, float)
    assert flops > 0.0


def test_calculate_mfu_returns_ratio():
    mfu = calculate_mfu(
        model_flops_per_step=2.0e12,
        step_time_seconds=1.0,
        device_peak_tflops=100.0,
        world_size=2,
    )

    assert mfu == 0.01


def test_build_mfu_log_uses_timer_and_accumulation_steps(monkeypatch):
    monkeypatch.setattr(openbee_mfu, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        openbee_mfu,
        "resolve_peak_tflops",
        lambda **kwargs: ("NVIDIA H200", 500.0),
    )

    logs = build_mfu_log(
        model=SimpleNamespace(
            calculate_model_flops=lambda **kwargs: 5.0e11,
        ),
        device_type="cuda",
        precision="bf16",
        batch_size=1,
        seq_len=128,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        step_time_seconds=2.0,
        gradient_accumulation_steps=4,
    )

    assert logs == {"perf/mfu": 0.002}


def test_normalize_device_name_maps_supported_variants():
    assert normalize_device_name("NVIDIA H200 NVL") == "NVIDIA H200"
    assert normalize_device_name("NVIDIA H100 PCIE 80GB") == "NVIDIA H100 PCIe"


def test_calculate_model_flops_freeze_reduces_flops():
    """Alignment stage (freeze_vit=True, freeze_llm=True) should produce fewer FLOPs
    than full training because frozen modules skip weight-grad backward passes."""
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                num_hidden_layers=2,
                hidden_size=16,
                intermediate_size=32,
                vocab_size=64,
            ),
            vision_config=SimpleNamespace(
                depth=3,
                hidden_size=8,
                intermediate_size=16,
                out_hidden_size=16,
                in_channels=3,
                patch_size=2,
                temporal_patch_size=1,
                spatial_merge_size=2,
            ),
        )
    )
    inject_model_flops_calculation(model)

    grid = torch.tensor([[1, 4, 4]])
    flops_full = model.calculate_model_flops(
        batch_size=1,
        seq_len=8,
        image_grid_thw=grid,
        is_training=True,
        freeze_vit=False,
        freeze_merger=False,
        freeze_llm=False,
    )
    # Alignment stage: only merger trains; ViT is 1×, LLM is 2×, merger is 3×
    flops_alignment = model.calculate_model_flops(
        batch_size=1,
        seq_len=8,
        image_grid_thw=grid,
        is_training=True,
        freeze_vit=True,
        freeze_merger=False,
        freeze_llm=True,
    )
    assert flops_alignment < flops_full

    # Inference (is_training=False) is unaffected by freeze flags
    flops_infer = model.calculate_model_flops(
        batch_size=1,
        seq_len=8,
        image_grid_thw=grid,
        is_training=False,
    )
    assert flops_infer < flops_alignment
