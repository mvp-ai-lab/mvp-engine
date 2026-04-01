# minimal-vlm

This recipe fine-tunes `Qwen/Qwen3-VL-2B-Instruct` on a local JSONL dataset.
The default demo dataset is at `data/minimal_vlm/demo.jsonl`.

`mvp_dataset` must be importable in the runtime environment. The recipe uses it
to shard local JSONL files under `.jsonl_shards/` next to the source dataset and
to build the training dataset.

## What it does

- Loads multimodal chat data from JSONL.
- Rewrites `<image>` placeholders into the Hugging Face chat format expected by Qwen3-VL.
- Supervises all assistant turns in the conversation.
- Freezes the visual stack by default and trains the language model plus `lm_head`.
- Runs through the shared distributed wrapper with a DDP-safe default mesh for `torchrun`.

## Dataset format

Each JSONL row must contain:

- `messages`: a non-empty list of chat messages with `role` and string `content`
- `images`: a list of image paths relative to the JSONL file

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

## Quick Start

Run the default demo config:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/minimal_vlm/configs/train.yaml
```

Point the recipe at a custom dataset:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/minimal_vlm/configs/train.yaml \
  data.train_path=/path/to/train.jsonl
```
