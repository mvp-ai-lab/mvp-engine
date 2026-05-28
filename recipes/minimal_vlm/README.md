# minimal-vlm

This recipe fine-tunes `Qwen/Qwen3-VL-2B-Instruct` on a local parquet dataset.
The default demo dataset is at `data/minimal_vlm/demo.parquet`.

`mvp_dataset` must be importable in the runtime environment. The recipe uses it
to build a resampled parquet training dataset.

This recipe intentionally does not use the kit layer in its implementation. It
is kept as the smallest explicit Qwen3-VL training example; kit-based MLLM
orchestration is demonstrated in `recipes/basic_vlm`.

## What it does

- Loads multimodal chat data from parquet.
- Rewrites `<image>` placeholders into the Hugging Face chat format expected by Qwen3-VL.
- Supervises all assistant turns in the conversation.
- Freezes the visual stack by default and trains the language model plus `lm_head`.
- Runs through the shared distributed wrapper with a DDP-safe default mesh for `torchrun`.

## Dataset format

Each parquet row must contain:

- `messages`: a non-empty list of chat messages with `role` and string `content`
- `images`: a list of image paths relative to the parquet data file

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

Download demo data:

```bash
hf download mvp-lab/mvp-engine-minimal-vlm-data \
  --repo-type dataset \
  --local-dir data/minimal_vlm
```

Run the default demo config:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/minimal_vlm/configs/train.yaml
```
