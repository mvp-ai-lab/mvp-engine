import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from model.vit import VIT_TP_MODULE_CONFIG, ViTForImageClassification, build_vit_model

from recipes.vit_classification.configs.schema import ViTModelConfig


def test_build_vit_model_uses_tp_enabled_wrapper():
    config = ViTModelConfig(
        pretrained_model_name_or_path="google/vit-base-patch16-224",
        load_pretrained_weights=False,
        num_classes=10,
        image_size=224,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
    )

    model = build_vit_model(config)

    assert isinstance(model, ViTForImageClassification)
    assert model.__class__.TP_MODULE_CONFIG == VIT_TP_MODULE_CONFIG


def test_vit_tp_module_config_matches_expected_runtime_class_names():
    assert VIT_TP_MODULE_CONFIG == {
        "ViTSelfAttention": {
            "query": "col",
            "key": "col",
            "value": "col",
        },
        "ViTSelfOutput": {
            "dense": "row",
        },
        "ViTIntermediate": {
            "dense": "col",
        },
        "ViTOutput": {
            "dense": "row",
        },
    }


def test_vit_tp_forward_uses_local_attention_head_metadata():
    project_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
    reference_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import os
        import sys
        import tempfile

        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        from torch.distributed.device_mesh import init_device_mesh

        sys.path.insert(0, {str(project_root)!r})
        sys.path.insert(0, {str(reference_root)!r})

        from mvp_engine.distributed.tp import parallelize_model_with_tensor_parallel
        from recipes.vit_classification.configs.schema import ViTModelConfig
        from model.vit import build_vit_model


        def worker(rank, world_size, init_file):
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{{init_file}}",
                rank=rank,
                world_size=world_size,
            )
            try:
                config = ViTModelConfig(
                    pretrained_model_name_or_path="google/vit-base-patch16-224",
                    load_pretrained_weights=False,
                    num_classes=10,
                    image_size=16,
                    patch_size=8,
                    num_channels=3,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    hidden_dropout_prob=0.0,
                    attention_dropout_prob=0.0,
                )

                model = build_vit_model(config)
                tp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tensor",))
                parallelize_model_with_tensor_parallel(model, tp_mesh)

                attention = model.vit.encoder.layer[0].attention.attention
                assert attention.num_attention_heads == 2, attention.num_attention_heads
                assert attention.all_head_size == 4, attention.all_head_size

                outputs = model(
                    pixel_values=torch.randn(2, 3, 16, 16),
                    labels=torch.zeros(2, dtype=torch.long),
                )
                assert outputs.logits.shape == (2, 10), outputs.logits.shape
            finally:
                dist.destroy_process_group()


        if __name__ == "__main__":
            with tempfile.NamedTemporaryFile(delete=False) as file_obj:
                init_file = file_obj.name

            try:
                mp.spawn(worker, args=(2, init_file), nprocs=2, join=True)
            finally:
                if os.path.exists(init_file):
                    os.unlink(init_file)
        """
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "run_vit_tp_forward.py"
        script_path.write_text(script)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

    combined_output = result.stdout + result.stderr
    if result.returncode != 0 and (
        "Operation not permitted" in combined_output or "Cannot resolve 127.0.0.1" in combined_output
    ):
        pytest.skip("Local Gloo process-group setup is not permitted in this environment.")

    assert result.returncode == 0, combined_output
