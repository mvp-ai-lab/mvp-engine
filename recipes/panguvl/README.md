# PanguVL

`recipes/panguvl` is an OpenPangu-VL training recipe built on top of the shared
`mvp_engine` launch, distributed, logging, and checkpoint infrastructure.

The recipe keeps OpenPangu model weights outside this repository. Only the
recipe code and a small set of patched OpenPangu runtime Python files are kept
here.

## What Is In This Recipe

- `configs/`: stage configs for alignment, pretraining, and SFT-style runs.
- `dataset/`: Open-Bee / multimodal chat preprocessing, label construction,
  packing, and collation.
- `engine/`: the recipe-local training engine.
- `model/`: model loading, freeze policy, loss patching, FLOPs estimation, and
  packed-attention helpers.
- `third_party/`: patched OpenPangu-VL runtime source files used with
  `trust_remote_code=True`.
- `tools/overlay_openpangu_code.py`: copies the patched runtime files into a
  local OpenPangu checkpoint directory.

## Why There Is A `third_party/` Directory

OpenPangu-VL is loaded through Hugging Face `trust_remote_code=True`, which
means the model imports Python files from the checkpoint directory itself. For
training this recipe, we need a patched copy of several OpenPangu runtime files.

This repository vendors only those runtime `.py` files:

```text
configuration_openpangu_vl.py
imageprocessor_openpangu_vl.py
modeling_openpangu_embedded.py
modeling_openpangu_vl.py
processor_openpangu_vl.py
videoprocessor_openpangu_vl.py
```

It does not vendor model weights or tokenizer artifacts. Users must obtain those
from the official OpenPangu-VL checkpoint.

Do not commit:

```text
model*.safetensors*
pytorch_model*.bin
tokenizer.model
tokenizer*.json
special_tokens_map.json
chat_template.json
generation_config.json
__pycache__/
*.pyc
```

## Step-By-Step Setup

Run these commands from the repository root.

### 1. Create The Project Environment

```bash
uv venv --python=3.12
uv sync
```

If the OpenPangu-VL Hugging Face repository requires authentication in your
environment, log in before downloading:

```bash
.venv/bin/hf auth login
```

### 2. Download The OpenPangu-VL Checkpoint

Choose a local directory outside the git-tracked recipe for model artifacts:

```bash
mkdir -p checkpoints
```

Download the official Hugging Face checkpoint:

```bash
.venv/bin/hf download FreedomIntelligence/openPangu-VL-7B \
  --local-dir checkpoints/OpenPangu-VL-7B
```

If your version of the Hugging Face CLI does not support `hf download`, use the
Python API instead:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FreedomIntelligence/openPangu-VL-7B",
    local_dir="checkpoints/OpenPangu-VL-7B",
)
PY
```

If you already downloaded the checkpoint somewhere else, move or symlink it into
a stable location:

```bash
mkdir -p checkpoints
ln -s /absolute/path/to/OpenPangu-VL-7B checkpoints/OpenPangu-VL-7B
```

### 3. Check The Downloaded Checkpoint

The checkpoint directory should contain the normal model and tokenizer
artifacts:

```bash
ls checkpoints/OpenPangu-VL-7B
```

At minimum, make sure these files exist:

```text
config.json
preprocessor_config.json
model.safetensors
model.safetensors.index.json
tokenizer.model
tokenizer_config.json
special_tokens_map.json
chat_template.json
```

### 4. Overlay The Patched OpenPangu Runtime Code

This recipe keeps patched OpenPangu runtime files under
`recipes/panguvl/third_party/`. Copy them into the downloaded checkpoint so
`trust_remote_code=True` loads the patched files:

```bash
.venv/bin/python recipes/panguvl/tools/overlay_openpangu_code.py \
  --checkpoint-dir checkpoints/OpenPangu-VL-7B \
  --check
```

If the check passes, apply the overlay:

```bash
.venv/bin/python recipes/panguvl/tools/overlay_openpangu_code.py \
  --checkpoint-dir checkpoints/OpenPangu-VL-7B
```

After overlay, verify that the checkpoint contains both the official artifacts
and the patched runtime files:

```bash
ls checkpoints/OpenPangu-VL-7B | grep -E 'modeling_openpangu|processor_openpangu|configuration_openpangu'
```

### 5. Point The Recipe At Your Dataset

Set the dataset path through a Hydra override when launching. For example:

```bash
DATA_PATH="/absolute/path/to/train/**/*.parquet"
```

The recipe expects Open-Bee style parquet rows with `messages` or
`conversations`, plus optional `images`.

### 6. Smoke-Test The Launch Configuration

Use one process and the debug config for a quick local run. The debug config
sets a small explicit `loop.total_steps`, so it skips the full dataset pass used
to infer training steps when `loop.total_steps=-1`.

```bash
torchrun --nproc_per_node=1 -m mvp_engine.launch \
  --config recipes/panguvl/configs/debug.yaml \
  model.pretrained_model_name_or_path=checkpoints/OpenPangu-VL-7B \
  data.train_path="$DATA_PATH"
```

For ad-hoc debugging with any stage config, set a positive step count, for
example `loop.total_steps=2`. Step inference only runs when
`loop.total_steps=-1`.

### 7. Run Distributed Training

Once the one-process run starts cleanly, launch the intended training job:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config recipes/panguvl/configs/stage1.yaml \
  model.pretrained_model_name_or_path=checkpoints/OpenPangu-VL-7B \
  data.train_path="$DATA_PATH"
```

## Dataset Format

The recipe expects parquet rows compatible with the Open-Bee style schema. Each
row should provide a conversation and optional images:

- `messages` or `conversations`: non-empty chat messages.
- `images`: image paths, bytes, or parquet image records.

Message roles are normalized from common formats such as `human`/`gpt` into
Hugging Face chat roles. Assistant turns are supervised; non-assistant tokens
are masked with `-100`.

The dataset code also supports:

- `<image>` placeholder expansion.
- OpenPangu/Qwen-style chat-template tokenization.
- optional thinking-tag handling through `data.enable_thinking`.
- optional sequence packing through `data.packing`.

## Updating The Vendored OpenPangu Code

When OpenPangu runtime changes are needed:

1. Edit or copy the patched `.py` files under `recipes/panguvl/third_party/`.
2. Keep weights and tokenizer artifacts out of this directory.
3. Run the overlay script against your local checkpoint.
4. Restart training so `trust_remote_code` reloads the checkpoint files.

If Hugging Face has already cached an older copy, restart the process. If needed,
clear the relevant cache under:

```text
~/.cache/huggingface/modules/transformers_modules/
```

## Notes For Contributors

- Keep generic training infrastructure in `mvp_engine/`.
- Keep PanguVL-specific behavior in `recipes/panguvl/`.
- Keep third-party OpenPangu runtime code isolated in
  `recipes/panguvl/third_party/`.
- Do not commit local data, outputs, checkpoints, model weights, or tokenizer
  artifacts.
