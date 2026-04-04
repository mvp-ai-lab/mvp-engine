# openbee

https://arxiv.org/html/2510.13795v4

`openbee` is planned as a three-stage VLM training workflow:

- alignment
- pretrain
- SFT

This initial recipe implements the alignment stage only. It is based on the
current `minimal_vlm` recipe shape and fine-tunes a local
`Qwen3-VL-8B-Instruct` checkpoint rebuilt for OpenBee on a local JSONL dataset.

The default placeholder dataset path is `data/openbee/alignment_demo.jsonl`.

`mvp_dataset` must be importable in the runtime environment. The recipe uses it
to shard local JSONL files under `.jsonl_shards/` next to the source dataset and
to build the training dataset.

## What it does

- Focuses on the alignment stage of the OpenBee training plan.
- Loads multimodal chat data from JSONL.
- Rewrites `<image>` placeholders into the Hugging Face chat format expected by Qwen3-VL.
- Supervises all assistant turns in the conversation.
- Freezes the visual stack by default and trains the language model plus `lm_head`.
- Uses a custom Qwen3-VL-8B checkpoint whose visual tower comes from
  `Qwen/Qwen3-VL-8B-Instruct`, language model and `lm_head` come from
  `Qwen/Qwen3-8B-Base`, and `visual.merger` is re-initialized randomly.
- Runs through the shared distributed wrapper with a DDP-safe default mesh for `torchrun`.

## Dataset format

Each JSONL row must contain:

- `messages`: a non-empty list of chat messages with `role` and string `content`
- `images`: a list of image paths relative to the JSONL file, or a list of
  `mvp_dataset` tar references such as `images/train-00000.tar#sample_0.jpg`

The total number of image paths must exactly match the total number of `<image>`
placeholders across the conversation.

Example:

```json
{
  "messages": [
    {"role": "user", "content": "<image>Who is this?"},
    {"role": "assistant", "content": "This is an example response."}
  ],
  "images": ["images/1.jpg"]
}
```

## Build Alignment Data

This recipe includes a converter that rebuilds the alignment corpus in the
recommended `mvp_dataset` JSONL + TAR-reference format from the downloaded
Open-Bee parquet releases:

```bash
python3 recipes/openbee/tools/build_alignment_data.py \
  --input-root /mnt/data-alpha-sg-01/team-camera/shared/mvp-engine/data/Open-Bee \
  --output-dir data/openbee/openbee_stage1_stage2_mm_only
```

It writes:

- `data/openbee/openbee_stage1_stage2_mm_only/train.jsonl`
- `data/openbee/openbee_stage1_stage2_mm_only/images/train-*.tar`

The current loader resolves those tar references automatically.

## Build The OpenBee 8B Checkpoint

Build the local checkpoint once before training:

```bash
.venv/bin/python recipes/openbee/tools/build_qwen3_vl_checkpoint.py \
  --output-dir recipes/openbee/pretrained/Qwen3-VL-8B-Instruct
```

The builder keeps the `Qwen/Qwen3-VL-8B-Instruct` model structure and visual
tower weights, swaps in `Qwen/Qwen3-8B-Base` for `language_model.*` and
`lm_head.*`, and randomly initializes `model.visual.merger.*`.

## Quick Start

Run the alignment-stage config:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/openbee/configs/train.yaml
```

Point the recipe at your alignment dataset:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/openbee/configs/train.yaml \
  data.train_path=/path/to/train.jsonl
```
