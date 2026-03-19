# minimal-vlm

This recipe fine-tunes `Qwen/Qwen3-VL-2B-Instruct` on the local demo dataset at `data/minimal_vlm/demo.jsonl`.

## What it does

- Loads multi-turn multimodal conversations from JSONL.
- Rewrites `<image>` placeholders into the Hugging Face chat format expected by Qwen3-VL.
- Builds supervised labels from assistant-token masks so all assistant turns contribute to loss.
- Freezes the visual stack by default and trains the language model plus `lm_head`.

## Dataset format

Each JSONL row must contain:

- `messages`: a list of chat messages with `role` and string `content`
- `images`: a list of image paths relative to the JSONL file

The total number of image paths must exactly match the total number of `<image>` placeholders across the conversation.

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

## Run

Launch with:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/minimal_vlm/configs/train.yaml
```

Key defaults in [`configs/train.yaml`](/home/c84391361/projects/mvp-engine/recipes/minimal_vlm/configs/train.yaml):

- `model.pretrained_model_name_or_path: Qwen/Qwen3-VL-2B-Instruct`
- `model.freeze_visual: true`
- `data.train_path: ./data/minimal_vlm/demo.jsonl`
- `data.eval_path: null` so evaluation dataset construction falls back to the training file
