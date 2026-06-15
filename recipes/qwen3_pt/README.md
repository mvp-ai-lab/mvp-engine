# Qwen3 Pretraining (qwen3_pt)

`qwen3_pt` is a text-only Qwen3 **pretraining** recipe. It follows the
`openbee` engine pattern with the multimodal/vision parts removed, and reuses
the generic `TokenNormedLossKit`, `OptimKit`, and `MFUKit` together with a new
text-only `LLMDataKit` / `LLMModelKit`.

## What it does

- Reads plain text from a Lance source via `mvp_dataset` (one configurable text
  field, default `data`).
- Tokenizes each document, appends EOS, and splits overlong documents into
  `max_seq_len` chunks (no data dropped by truncation).
- Packs short documents into longer sequences and isolates them with
  `pack_segment_ids` + a block-causal mask.
- Trains with **full-token next-token loss** (`labels = input_ids`), but masks
  the loss at every document boundary so the last token of one document is not
  trained to predict the first token of the next.
- Uses token-normalized loss across gradient accumulation and data-parallel
  ranks (`TokenNormedLossKit`), FSDP2/DDP wrapping, gradient checkpointing,
  optional `torch.compile`, and MFU logging.
- Supports both random-initialized from-scratch pretraining
  (`model.train_from_scratch=true`, `model.init_seed`) and continued
  pretraining from `Qwen/Qwen3-*-Base` weights (`train_from_scratch=false`).

## Data

The Lance source should expose a text column (default `data`). Each row is one
raw document.

## Run

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/qwen3_pt/configs/train.yaml
```

Override for real training, for example:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/qwen3_pt/configs/train.yaml \
  data.train_path=./data/<lance-dataset>/meta.json \
  model.pretrained_model_name_or_path=Qwen/Qwen3-8B-Base
```

`train.yaml` defaults to SDPA attention. FlashAttention 2 for packed text is not
yet validated (see `model/packing/prepare.py`); keep `attn_implementation=sdpa`
until the FA2 padding path is verified.

## Notes

- This recipe does NOT pin the transformers version; it uses the engine's
  current install (verified to support Qwen3). It is therefore a *better-result*
  experiment, not a bit-exact reproduction of any external baseline.
- Default `model.init_seed=42` matches the seed used by the external baseline,
  but exact random-init parity is not guaranteed across library versions.
