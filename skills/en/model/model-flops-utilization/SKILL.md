---
name: model-flops-utilization
description: Implement end-to-end MFU support for the current model and engine, including model FLOPs estimation, hardware peak FLOPs lookup, runtime MFU calculation, and MFU logging. Use when MFU must be added to a real training workflow instead of stopping at FLOPs formulas.
---

# Model FLOPs Utilization

> 中文版：`skills/zh-cn/model/model-flops-utilization/SKILL.md`
> English version: `skills/en/model/model-flops-utilization/SKILL.md`

## Goal

- Add `calculate_model_flops(...) -> float` to the current model instance with an injection helper instead of replacing the model class.
- Add or wire `calculate_mfu(...) -> float` in the engine so MFU is computed during training from model FLOPs, step time, peak device FLOPs, precision, and `world_size`.
- Detect the current hardware, look up peak FLOPs from `references/hardware_peak_flops.csv`, and add MFU into the training metrics dict with key `perf/mfu`.
- If the current environment is CPU-only, first search local `*.md` files for GPU or cluster instructions and follow them before continuing. If none exist, ask the user how to run on GPU.

## Required Inputs

- The model entrypoint that creates the runtime model instance.
- The engine or training loop location where step time and logger access are available.
- The real model architecture: ViT, decoder-only Transformer, seq2seq encoder-decoder, or VLM composed from vision and language stacks.
- Runtime facts needed for MFU:
  - training precision such as `bf16`, `fp16`, or `fp32`
  - `world_size`
  - step time in seconds or an equivalent timing source already tracked by the engine
  - current device name from `nvidia-smi` or user input
- The current working directory, because CPU-only fallback requires scanning local `*.md` files for GPU training instructions.

## Workflow

### 0. Review reference implementation pattern first

- If `references/recipes/` contains an MFU example, treat it as an implementation sample first. Use it to understand which files and code layers usually need changes before editing the current recipe.
- The sample is only for pattern recognition. Do not assume the current model, field names, config hierarchy, or training flow match the sample exactly.
- When comparing the sample with the current recipe, focus on these questions:
  - how model FLOPs are attached to the model instance
  - where MFU configuration enters the schema or config
  - where `mfu` is computed and logged after a training step
  - where `world_size`, step time, and peak hardware FLOPs come from
- Reuse the sample's class names, file names, or field names only when they are compatible with the current recipe. Otherwise, adapt them to the current model and engine structure.
- If the sample and the current recipe differ, prefer extracting the implementation pattern and data flow over copying code verbatim.
- When citing demo integration flow, point to `references/recipes/vit_classification_addon/model/vit.py`, `references/recipes/vit_classification_addon/engine/vit_classification_engine.py`, `references/recipes/vit_classification_addon/configs/schema.py`, and `references/recipes/vit_classification_addon/configs/train.yaml`.

### 1. Detect the runtime environment first

- Check whether training is already running on GPU.
- If GPU is available, identify the device name with `nvidia-smi` and use that for peak FLOPs lookup.
- If the current environment is CPU-only:
  - scan `*.md` files under the current working directory for GPU, Slurm, cluster, `torchrun`, or launch instructions
  - if instructions exist, follow them and move the workflow to a GPU-capable run
  - if no instructions exist, pause and ask the user how GPU training should be launched
- Ask the user only when hardware identity, precision, or launch method cannot be derived reliably from the environment.

### 2. Add `calculate_model_flops(...)` to the current model instance

- Do not replace the model class with a subclass as the default pattern.
- Add a helper named `inject_model_flops_calculation(model)` and attach `calculate_model_flops` to the existing instance with `types.MethodType`.
- Keep only the smallest explicit parameter set required by the real architecture.
- Do not include unused parameters in the generated method signature.
- Return one `float`: model FLOPs per training or eval step for the current process.

Use this injection pattern:

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, ..., is_training: bool = True) -> float:
        ...

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model


model = AutoModel.from_pretrained(...)
model = inject_model_flops_calculation(model)
```

#### ViT example

Use this shape when the model is a patch-embedding vision transformer.

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        is_training: bool = True,
    ) -> float:
        if isinstance(image_size, int):
            image_h, image_w = image_size, image_size
        else:
            image_h, image_w = map(int, image_size)
        if isinstance(patch_size, int):
            patch_h, patch_w = patch_size, patch_size
        else:
            patch_h, patch_w = map(int, patch_size)

        batch = int(batch_size)
        if min(batch, image_h, image_w, patch_h, patch_w) <= 0:
            raise ValueError("batch_size, image_size, and patch_size must be > 0")
        if image_h % patch_h != 0 or image_w % patch_w != 0:
            raise ValueError("image_size must be divisible by patch_size")

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        channels = int(getattr(self.config, "num_channels", 3))
        hidden = int(self.config.hidden_size)
        layers = int(self.config.num_hidden_layers)
        intermediate = int(self.config.intermediate_size)
        num_labels = int(getattr(self.config, "num_labels", 1000))

        patch_embed_flops = 2 * batch * num_patches * (channels * patch_h * patch_w) * hidden
        block_flops = (
            8 * batch * num_patches * hidden * hidden
            + 4 * batch * num_patches * num_patches * hidden
            + 4 * batch * num_patches * hidden * intermediate
        )
        head_flops = 2 * batch * hidden * num_labels
        forward_flops = float(patch_embed_flops + layers * block_flops + head_flops)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### Decoder-only Transformer example

Use this shape when the model is a dense decoder-only language model.

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        seq_len: int,
        is_training: bool = True,
    ) -> float:
        batch = int(batch_size)
        tokens = int(seq_len)
        if batch <= 0 or tokens <= 0:
            raise ValueError("batch_size and seq_len must be > 0")

        layers = int(getattr(self.config, "num_hidden_layers", self.config.n_layer))
        hidden = int(getattr(self.config, "hidden_size", self.config.n_embd))
        intermediate = int(getattr(self.config, "intermediate_size", 4 * hidden))
        vocab = int(self.config.vocab_size)

        per_layer = (
            8 * batch * tokens * hidden * hidden
            + 4 * batch * tokens * tokens * hidden
            + 4 * batch * tokens * hidden * intermediate
        )
        lm_head_flops = 2 * batch * tokens * hidden * vocab
        forward_flops = float(layers * per_layer + lm_head_flops)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```


#### Seq2seq encoder-decoder example

Use this for T5/BART-style models. Pass separate `encoder_seq_len` and `decoder_seq_len`, and include decoder cross-attention FLOPs.

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        encoder_seq_len: int,
        decoder_seq_len: int,
        is_training: bool = True,
    ) -> float:
        b = int(batch_size)
        s_enc = int(encoder_seq_len)
        s_dec = int(decoder_seq_len)
        if min(b, s_enc, s_dec) <= 0:
            raise ValueError("batch_size, encoder_seq_len, decoder_seq_len must be > 0")

        cfg = self.config
        hidden = int(cfg.d_model)
        ffn = int(cfg.d_ff)
        enc_layers = int(getattr(cfg, "num_layers", cfg.num_hidden_layers))
        dec_layers = int(getattr(cfg, "num_decoder_layers", enc_layers))
        vocab = int(cfg.vocab_size)

        enc_layer = 8 * b * s_enc * hidden * hidden + 4 * b * s_enc * s_enc * hidden + 4 * b * s_enc * hidden * ffn
        dec_self = 8 * b * s_dec * hidden * hidden + 4 * b * s_dec * s_dec * hidden + 4 * b * s_dec * hidden * ffn
        dec_cross = 4 * b * s_dec * s_enc * hidden
        lm_head = 2 * b * s_dec * hidden * vocab

        forward_flops = float(enc_layers * enc_layer + dec_layers * (dec_self + dec_cross) + lm_head)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### VLM example

Use this pattern when the runtime model combines a vision encoder and a decoder-only language model. Do not assume every VLM uses the same field names.

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        seq_len: int,
        is_training: bool = True,
    ) -> float:
        vision_flops = ...
        language_flops = ...
        forward_flops = float(vision_flops + language_flops)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

- For VLMs, include both the vision-side and language-side compute.
- Adapt field names such as `vision_config`, `text_config`, `hidden_size`, and `num_hidden_layers` to the real implementation instead of copying template names blindly.

### 3. Implement MFU in the engine

- Add or reuse an engine-side helper named `calculate_mfu(...)`.
- MFU must be computed from runtime throughput, not from model FLOPs alone.
- Prefer the real step timing source already used by the engine. If the engine already tracks iteration duration, reuse it instead of adding a parallel timer.

Use this formula unless the current engine already has a stronger established convention:

```python
def calculate_mfu(
    *,
    model_flops_per_step: float,
    step_time_seconds: float,
    device_peak_tflops: float,
    world_size: int,
) -> float:
    if step_time_seconds <= 0:
        raise ValueError("step_time_seconds must be > 0")
    if device_peak_tflops <= 0:
        raise ValueError("device_peak_tflops must be > 0")
    if world_size <= 0:
        raise ValueError("world_size must be > 0")

    total_peak_flops = device_peak_tflops * 1e12 * world_size
    achieved_flops_per_second = model_flops_per_step / step_time_seconds
    return float(achieved_flops_per_second / total_peak_flops)
```

- `model_flops_per_step` should use the model method from the current process perspective.
- `device_peak_tflops` is the single-device peak for the active precision.
- `world_size` must reflect the number of training devices that contribute to the step.
- If gradient accumulation or pipeline parallel timing changes the effective step definition in the current engine, adapt the numerator and timing source to that real step boundary and state the assumption in code comments.

### 4. Resolve hardware peak FLOPs

- First try to detect the active GPU with `nvidia-smi`.
- Normalize the detected device name to the closest row in `references/hardware_peak_flops.csv`.
- Match the active precision such as `bf16` or `fp16`.
- If the device cannot be matched reliably, ask the user which hardware and precision should be used for MFU.
- Keep the lookup logic simple and explicit. Do not hide a fallback that silently picks the wrong GPU row.

### 5. Write MFU into the metrics dict

- Compute MFU inside the engine where step timing and logger access already exist.
- Add MFU into the training metrics dict in the same place where the engine reports step metrics.
- Use explicit metric key: `perf/mfu`.

Example:

```python
mfu = self.calculate_mfu(
    model_flops_per_step=model_flops,
    step_time_seconds=step_time,
    device_peak_tflops=device_peak_tflops,
    world_size=world_size,
)
log_dict["perf/mfu"] = float(mfu)
logger.log(log_dict, step=step)
```

## Validation and Acceptance Checklist

- The current model instance has `calculate_model_flops(...)` and the method was added by injection instead of replacing the model class.
- `calculate_model_flops(...)` keeps only the parameters required by the current architecture.
- The engine has `calculate_mfu(...)` or an equivalent MFU integration point.
- MFU uses runtime step time, single-device peak FLOPs, active precision, and `world_size`.
- The MFU value written to logs is a `float`.
- Training logs include `perf/mfu`.
- The logged MFU value passes a sanity check:
  - it is not negative
  - it is not implausibly larger than `1`
  - if it is suspicious, re-check model FLOPs, hardware lookup, precision mapping, timing source, and `world_size`

## Output

- Summarize where `calculate_model_flops(...)` was injected.
- Summarize where engine-side MFU was added or updated.
- State which GPU and precision were used for peak FLOPs lookup.
- State whether the current run used single-GPU or multi-GPU assumptions.
- State where `mfu=...` is logged.
- If GPU access was not available and no local launch instructions were found, state that user input is still required before MFU can be completed safely.

## Read On Demand

- Read `references/hardware_peak_flops.csv` when you need a peak FLOPs lookup for a known GPU and precision.
