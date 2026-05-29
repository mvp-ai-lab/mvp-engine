# Video VLM

`video_vlm` is the video-capable VLM recipe for Qwen3-VL training. It mirrors
`recipes/basic_vlm/` for the engine, packing, loss guard, optimizer, scheduler,
checkpointing, and distributed launch flow, while changing the vision path:

- images are processed with the OneVision image processor;
- videos use the OneVision encoder tower plus on-the-fly codec patchification;
- video batches carry `pixel_values_videos`, `video_grid_thw`, and
  `patch_positions`.

The first demo path uses tiny Hugging Face viewer parquet files converted into
the local `video_vlm` schema. A second codec smoke path generates synthetic
H.264 and H.265 clips to validate real `cv_reader` residual extraction.

## Prerequisites

- Work from the `mvp-engine` repo with the project virtualenv active:

  ```bash
  cd /path/to/mvp-engine
  source .venv/bin/activate
  ```

- Make sure the Qwen3-VL checkpoint configured by
  `model.pretrained_model_name_or_path` is accessible. The demo configs default
  to `Qwen/Qwen3-VL-2B-Instruct`.

- Make sure the OneVision encoder model configured by
  `model.vision_encoder_name_or_path` is accessible. The default is
  `lmms-lab-encoder/onevision-encoder-large`, which Hugging Face can download on
  first use. To prefetch it:

  ```bash
  hf download lmms-lab-encoder/onevision-encoder-large
  ```

- Install `cv_reader` for real codec residual extraction:

  ```bash
  bash batch_files/install_cv_reader.sh
  .venv/bin/python -c "from cv_reader import api; print(api)"
  ```

  The helper installs from
  `${ONEVISION_ENCODER_ROOT}/llava_next/Compressed_Video_Reader`, where
  `ONEVISION_ENCODER_ROOT` should point at your local OneVision-Encoder clone.
  The manual equivalent is:

  ```bash
  source .venv/bin/activate
  export ONEVISION_ENCODER_ROOT=/path/to/OneVision-Encoder
  cd "$ONEVISION_ENCODER_ROOT/llava_next/Compressed_Video_Reader"
  bash install.sh
  python -c "from cv_reader import api; print(api)"
  ```

  The unified `cv_reader` path currently supports real codec residuals for
  H.264/AVC (`h264`) and H.265/HEVC (`hevc`/`h265`). Other codecs may still
  decode as RGB video, but they do not provide true codec residuals through this
  recipe.

## Stages

- `configs/stage1.yaml`: alignment stage.
- `configs/stage2.yaml`: pretraining stage.
- `configs/stage3.yaml`: SFT stage.
- `configs/demo.yaml`: tiny Hugging Face viewer demo.
- `configs/hevc_smoke.yaml`: tiny synthetic H.264/H.265 codec residual smoke.

The three training stages keep the same shape as `basic_vlm`: they expect a
dataset path and a model/checkpoint path, and they run through
`mvp_engine.launch`. Override `data.train_path` and
`model.pretrained_model_name_or_path` for your data and checkpoints.

## Data

Rows should provide:

- `messages` or `conversations`: user/assistant chat turns.
- `images`: image references consumed by `<image>` placeholders.
- `image_size` or `img_size`: image size metadata matching `images`.
- `videos` or `video`: video references consumed by `<video>` placeholders.

The loader validates raw rows, tokenizes conversations with the Qwen3-VL
tokenizer/template, preprocesses images with the OneVision image processor, and
materializes video tensors by selecting codec-salient patches on the fly.

### Hugging Face Viewer Demo

The default demo data is written to:

```text
/path/to/data/video_vlm/demo/*.parquet
```

It is generated from selected preview parquet files from
`mvp-lab/LLaVA-OneVision-2-Data`, such as `viewer/spatial.parquet` and
`viewer/caption_gt10min.parquet`. These preview files are useful for schema and
training smoke tests because they are small and include image/video rows.

Download only the preview files:

```bash
TARGET_DIR="/path/to/data/LLaVA-OneVision-2-Data-viewer"
DATASETNAME="mvp-lab/LLaVA-OneVision-2-Data"

hf download "$DATASETNAME" \
  --repo-type=dataset \
  --local-dir "$TARGET_DIR" \
  --max-workers 4 \
  viewer/spatial.parquet \
  viewer/caption_gt10min.parquet
```

Convert a tiny local demo parquet:

```bash
.venv/bin/python recipes/video_vlm/tools/convert_llava_onevision_viewer.py \
  --input-dir "$TARGET_DIR" \
  --output-dir /path/to/data/video_vlm/demo \
  --image-limit 2 \
  --video-limit 1
```

The HF viewer demo is the recommended first end-to-end training smoke. It is not
the strongest proof of codec residual extraction, because the downloaded preview
video codec may vary.

### Codec Smoke Data

Use the codec smoke utility when you specifically need to prove the real
`cv_reader` path:

```bash
.venv/bin/python recipes/video_vlm/tools/create_hevc_smoke_dataset.py
```

It writes:

- `synthetic_h264.mp4`
- `synthetic_h265.mp4`
- `codec_smoke.parquet`

under:

```text
/path/to/data/video_vlm/codec_smoke
```

Run it with `configs/hevc_smoke.yaml` and `data.cv_reader_required=true` to fail
loudly if real codec residuals are unavailable.

## Codec Config

Video-specific config fields live under `data` and `model`:

- `data.video_placeholder`: text marker replaced by video visual tokens.
- `data.codec_enabled`: enables codec patchification for video rows.
- `data.codec_num_frames`: number of source frames sampled from each video.
- `data.codec_packed_frames`: number of dense packed frames sent to OneVision.
- `data.codec_frame_size`: square frame size used by codec packing.
- `data.codec_patch_size`: spatial patch size for residual top-K selection.
- `data.codec_k_keep`: number of patches kept; must equal
  `codec_packed_frames * (codec_frame_size / codec_patch_size) ** 2`.
- `data.cv_reader_required`: when true, real H.264/H.265 residuals are required;
  when false, the recipe may fall back to frame-difference residuals.
- `data.hevc_decoder_bin`: legacy/backward-compatible field; the unified
  H.264/H.265 path uses `cv_reader` instead.
- `model.vision_encoder_name_or_path`: OneVision encoder HF model or local path.
- `model.freeze_vision_encoder`: freezes the OneVision encoder tower.

Default demo geometry is 64 sampled source frames, 224x224 frames, 14x14
patches, 2048 kept patches, and 8 packed codec frames. The codec smoke config
uses a smaller 8-frame, 1-packed-frame setup for speed.

## Run

HF viewer demo:

```bash
CONFIG=./recipes/video_vlm/configs/demo.yaml sbatch batch_files/run_video_vlm_demo.sh
```

Codec residual smoke:

```bash
CONFIG=./recipes/video_vlm/configs/hevc_smoke.yaml sbatch batch_files/run_video_vlm_demo.sh
```

Direct local/debug launch:

```bash
torchrun \
  --nproc_per_node=1 \
  --master_port=29501 \
  -m mvp_engine.launch \
  --config ./recipes/video_vlm/configs/demo.yaml
```

Training stages:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/video_vlm/configs/stage1.yaml
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/video_vlm/configs/stage2.yaml
torchrun --nproc_per_node=8 -m mvp_engine.launch --config ./recipes/video_vlm/configs/stage3.yaml
```

Example override:

```bash
torchrun --nproc_per_node=8 -m mvp_engine.launch \
  --config ./recipes/video_vlm/configs/stage2.yaml \
  data.train_path=/path/to/video_vlm/train/*.parquet \
  model.pretrained_model_name_or_path=/path/to/Qwen3-VL-checkpoint
```

## Tests

Focused codec and preprocessing tests:

```bash
.venv/bin/pytest \
  recipes/video_vlm/tests/test_codec.py \
  recipes/video_vlm/tests/test_preprocess_onevision.py \
  -q
```

Broader recipe checks:

```bash
.venv/bin/pytest recipes/video_vlm/tests -q
```

The codec tests cover deterministic residual top-K selection, flattened index to
`[t, h, w]` conversion, packed frame shape, placeholder/token alignment, mocked
`cv_reader` routing, and fallback behavior. A successful codec smoke should
produce a packed video tensor, `video_grid_thw`, and `patch_positions` whose
length matches `data.codec_k_keep`.
