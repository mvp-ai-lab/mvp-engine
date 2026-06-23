# Qwen3

`qwen3` is a text-only Qwen3 pretrain recipe. It uses the generic `LLMDataKit` /
`LLMModelKit` together with `TokenNormedLossKit`, `OptimKit`, and `MFUKit`.

## What it does

- Reads plain text from a Lance source via `mvp_dataset` (one configurable text
  field, default `data`).
- Tokenizes each document, appends EOS, and splits overlong documents into
  `max_seq_len` chunks (no data dropped by truncation).
- Packs tokenized samples with the LLM Kit stream packer, concatenating tokens
  in order and slicing fixed-length training chunks.
- Trains with **full-token next-token loss** (`labels = input_ids`) on the
  packed stream. Sample-level attention/position isolation is available through
  config, but disabled by default.
- Uses token-normalized loss across gradient accumulation and data-parallel
  ranks (`TokenNormedLossKit`), FSDP2/DDP wrapping, gradient checkpointing,
  optional `torch.compile`, and MFU logging.
- Supports both random-initialized pretraining
  (`model.load_pretrained_model=false`, seeded by top-level `seed`) and
  continued pretraining from `Qwen/Qwen3-*-Base` weights
  (`load_pretrained_model=true`).

## Data

The Lance source should expose a text column (default `data`). Each row is one
raw document.

## Run

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/qwen3/configs/train.yaml
```

Override for real training, for example:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/qwen3/configs/train.yaml \
  data.train_path=./data/<lance-dataset>/meta.json \
  model.pretrained_model_name_or_path=Qwen/Qwen3-8B-Base
```

`train.yaml` defaults to FlashAttention 2. Packed text batches pass varlen
sequence metadata to FA2; set `model.attn_implementation=sdpa` or `eager` to use
the 4D block-causal mask fallback.

## Notes

- This recipe does NOT pin the transformers version; it uses the engine's
  current install (verified to support Qwen3). It is therefore a *better-result*
  experiment, not a bit-exact reproduction of any external baseline.
- Default `seed=42` is used for random model initialization, but exact
  random-init parity is not guaranteed across library versions.
