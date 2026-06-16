# Interleaved

`interleaved` is a Qwen3-VL recipe for image-text interleaved training. It
ports the main settings from
`LLaMA-Factory-private/projects/interleaved` onto MVP Engine's packed MLLM
training stack.

## Data

The recipe uses `MLLMDataKit` with recipe-local `InterleavedSampleKit` and
`InterleavedMediaKit`.

Supported rows:

- ShareGPT/OpenBee rows with `messages` or `conversations`, `images`, and
  `image_size` or `img_size`.
- OpenAI content-block rows with `messages[].content` blocks such as
  `{"type": "text", "text": "..."}` and
  `{"type": "image", "image_file": {"image": ...}}`.
- LLaMA-Factory converted rows with `_response` and `_images`.

For OpenAI content-block rows that contain a single user message, the adapter
uses the LLaMA-Factory interleaved convention: an empty user prompt followed by
one assistant response containing the interleaved text and images.

The current `mvp_dataset.Dataset` API has no runtime dataset-mixing primitive.
Configs that mirror `dataset="wiki,bee"` therefore point to a pre-merged Lance
`meta.json`. Build that merged dataset with the intended sampling ratios before
launching.

## Configs

- `wiki_bee_stage2_64k.yaml`: main wiki+Bee 64k full-SFT setting.
- `debug_wiki_64k.yaml`: short 64k debug run.
- `debug_wiki_cut.yaml`: 16k debug variant from the reference project.
- `wiki_bee_scaledown_stage2_64k.yaml`: scaledown stage-2 variant.
- `wiki_merge_missing0420_stage2_64k.yaml`: merged wiki stage-2 variant.

## Run

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/interleaved/configs/wiki_bee_stage2_64k.yaml
```

Common overrides:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/interleaved/configs/wiki_bee_stage2_64k.yaml \
  data.train_path=/path/to/merged/meta.json \
  model.pretrained_model_name_or_path=/path/to/qwen3-vl-checkpoint
```
