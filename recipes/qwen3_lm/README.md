# Qwen3 LM

`qwen3_lm` is a text-only Qwen3 supervised fine-tuning recipe. It follows the
`basic_vlm` training pattern while removing multimodal-specific image handling.

## What it does

- Loads chat-style text data from local JSONL or an `mvp_dataset` source.
- Uses iteration-based infinite training with resampling by default.
- Tokenizes multiturn chat data with the Qwen3 chat template.
- Supports Qwen3 thinking-mode policies for empty or non-empty `<think>` blocks.
- Optionally packs short samples into longer sequences.
- Uses unreduced per-token loss and global supervised-token normalization across
  gradient accumulation and data-parallel ranks.
- Supports FlashAttention 2 packed segment masks, gradient checkpointing,
  `torch.compile`, FSDP2/DDP wrapping, and MFU logging.

## Dataset

Each JSONL row should provide either `messages` or `conversations`.

```json
{"messages":[
  {"role":"system","content":"You are helpful."},
  {"role":"user","content":"Where is Hangzhou?"},
  {"role":"assistant","content":"<think>\n...\n</think>\n\nHangzhou is in Zhejiang, China."}
]}
```

Rows with pre-tokenized `input_ids`, `attention_mask`, and optional `labels` are
also accepted for smoke tests and debugging.

## Run

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/qwen3_lm/configs/train.yaml
```

Override the model and dataset paths for real training:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/qwen3_lm/configs/train.yaml \
  data.source_type=mvp_dataset \
  data.source_format=lance \
  data.train_path=./data/Qwen3-LM-Lance/meta.json \
  model.pretrained_model_name_or_path=./pretrained/Qwen3-8B
```

`train.yaml` follows `basic_vlm` and uses FlashAttention 2 by default. On this
cluster, `flash-attn==2.8.3` was built on a Slurm H200 node after loading
`cuda/12.8`. If the environment has no compatible `flash_attn` install, use
`model.attn_implementation=sdpa`.

## Validation

This recipe was validated on H200 Slurm nodes with:

- tiny 1-GPU smoke training
- 2-GPU `mvp_dataset` JSONL + packing smoke training
- real Qwen3-8B, 2-GPU FSDP2, bf16, SDPA, gradient checkpointing, packing
- real Qwen3-8B, 2-GPU FSDP2, bf16, FlashAttention 2, gradient checkpointing, packing
