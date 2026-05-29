"""Focused packing tests for the Qwen3 LM recipe."""

from types import SimpleNamespace

import torch

from recipes.qwen3_lm.dataset.dataset import build_jsonl_worker_order
from recipes.qwen3_lm.model.packing.fa2_patch import apply_packed_fa2_patch
from recipes.qwen3_lm.model.packing.prepare import _build_packed_block_causal_mask
from recipes.qwen3_lm.model.qwen3 import inject_model_flops_calculation


def test_jsonl_worker_order_partitions_one_shared_shuffle():
    """Validate worker shards cover a JSONL round exactly once."""
    shards = [
        build_jsonl_worker_order(
            10,
            seed=42,
            round_index=0,
            worker_id=worker_id,
            num_workers=2,
            rank=0,
            world_size=1,
        )
        for worker_id in range(2)
    ]

    merged = shards[0] + shards[1]
    assert sorted(merged) == list(range(10))
    assert len(set(merged)) == 10


def test_jsonl_worker_order_keeps_small_infinite_datasets_live():
    """Validate small JSONL datasets still feed every distributed worker."""
    shards = [
        build_jsonl_worker_order(
            3,
            seed=42,
            round_index=0,
            worker_id=0,
            num_workers=1,
            rank=rank,
            world_size=8,
        )
        for rank in range(8)
    ]

    assert all(len(shard) == 1 for shard in shards)
    assert all(0 <= shard[0] < 3 for shard in shards)


def test_packed_eager_mask_blocks_cross_segment_attention():
    """Validate packed 4D masks isolate segments and keep causal order."""
    pack_segment_ids = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
    attention_mask = _build_packed_block_causal_mask(pack_segment_ids, dtype=torch.float32)

    assert attention_mask.shape == (1, 1, 5, 5)
    assert attention_mask[0, 0, 0, 0] == 0
    assert attention_mask[0, 0, 1, 0] == 0
    assert attention_mask[0, 0, 0, 1] < -1e30
    assert attention_mask[0, 0, 2, 0] < -1e30
    assert attention_mask[0, 0, 2, 2] == 0
    assert attention_mask[0, 0, 3, 2] == 0
    assert attention_mask[0, 0, 4, 4] < -1e30


def test_fa2_patch_covers_dense_qwen3_and_qwen3_moe():
    """Validate packed FA2 patch updates both dense and MoE Qwen3 modules."""
    segment_mask = torch.tensor([[1, 1, 2]], dtype=torch.long)
    apply_packed_fa2_patch()

    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_mod

    assert qwen3_mod.create_causal_mask(attention_mask=segment_mask) is segment_mask
    assert qwen3_moe_mod.create_causal_mask(attention_mask=segment_mask) is segment_mask
    assert qwen3_moe_mod.create_sliding_window_causal_mask(attention_mask=segment_mask) is segment_mask


def test_moe_flops_estimator_handles_sparse_config():
    """Validate FLOPs accounting handles Qwen3-MoE-specific config fields."""
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        decoder_sparse_step=1,
        moe_intermediate_size=16,
        num_experts_per_tok=2,
        num_experts=4,
    )
    model = inject_model_flops_calculation(SimpleNamespace(config=config))

    flops = model.calculate_model_flops(batch_size=1, seq_len=8, attention_mask=torch.ones(1, 8), is_training=True)

    assert flops > 0
