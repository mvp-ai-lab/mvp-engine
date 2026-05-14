---
name: model-flops-utilization
description: Implement end-to-end MFU support for the current model and engine, including model FLOPs estimation, hardware peak FLOPs lookup, runtime MFU calculation, and MFU logging.
---

# Model FLOPs Utilization

## Goal

- Add `calculate_model_flops(...) -> float` to the current model instance by injection instead of replacing the model class.
- Add or wire `calculate_mfu(...) -> float` in the engine so MFU is computed from runtime step timing, model FLOPs, active precision, peak device FLOPs, and `world_size`.
- Log MFU in the training metrics dict under `perf/mfu`.
- Use the real runtime environment and hardware identity rather than hard-coded assumptions.

## Required Inputs

- The entrypoint that creates the runtime model instance.
- The engine or training loop location where step timing and logger access already exist.
- The real model architecture, such as ViT, decoder-only Transformer, encoder-decoder, or a VLM composed from vision and language stacks.
- Runtime facts required for MFU:
  - precision such as `bf16`, `fp16`, or `fp32`
  - `world_size`
  - step time in seconds, or an equivalent timing source already tracked by the engine
  - the active device name from `nvidia-smi` or user input
- The current working directory, because CPU-only fallback requires scanning local `*.md` files for GPU launch instructions.

## Workflow

### 1. Review the reference pattern first

- If `references/recipes/` contains an MFU example, treat it as a pattern sample instead of something to copy verbatim.
- Focus on these questions:
  - how model FLOPs are attached to the runtime model instance
  - where MFU enters schema or config
  - where the engine computes and logs `mfu`
  - where timing, `world_size`, and peak hardware FLOPs come from
- Prefer extracting the data flow and integration points over reusing sample names mechanically.

### 2. Detect the runtime environment before editing

- Check whether the current workflow already runs on GPU.
- If a GPU is available, use `nvidia-smi` to identify the active device for peak-FLOPs lookup.
- If the current environment is CPU-only:
  - scan local `*.md` files for GPU, Slurm, cluster, `torchrun`, or launch instructions
  - follow those instructions if they exist
  - otherwise, stop and ask the user how GPU training should be launched
- Ask the user only when hardware identity, precision, or launch method cannot be derived reliably.

### 3. Inject `calculate_model_flops(...)` into the current model instance

- Do not replace the model class with a subclass as the default pattern.
- Add an `inject_model_flops_calculation(model)` helper and bind `calculate_model_flops` with `types.MethodType`.
- Keep only the minimum parameter set required by the actual architecture.
- Return a single `float` representing model FLOPs per step for the current process.

Use this base injection pattern:

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, ..., is_training: bool = True) -> float:
        ...

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### ViT example

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        is_training: bool = True,
    ) -> float:
        batch = int(batch_size)
        if batch <= 0:
            raise ValueError("batch_size must be > 0")

        image_size = int(self.config.image_size)
        patch_size = int(self.config.patch_size)
        num_patches = (image_size // patch_size) ** 2
        channels = int(getattr(self.config, "num_channels", 3))
        hidden = int(self.config.hidden_size)
        layers = int(self.config.num_hidden_layers)
        intermediate = int(self.config.intermediate_size)
        num_labels = int(getattr(self.config, "num_labels", 1000))

        patch_embed_flops = 2 * batch * num_patches * (channels * patch_size * patch_size) * hidden
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

#### Encoder-decoder example

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

### 4. Implement MFU in the engine

- Add or reuse an engine-side helper named `calculate_mfu(...)`.
- Reuse the real step timing source already tracked by the engine instead of adding a parallel timer when possible.
- Use runtime throughput, not static model FLOPs alone.

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

- If gradient accumulation or pipeline timing changes the effective step boundary, adapt the numerator and timing source to that real boundary and document the assumption.

### 5. Resolve hardware peak FLOPs explicitly

- Detect the active GPU with `nvidia-smi` when possible.
- Normalize the detected device name to the closest row in `references/hardware_peak_flops.csv`.
- Match the active precision such as `bf16` or `fp16`.
- If the device cannot be matched reliably, stop and ask the user which hardware and precision should drive MFU.
- Keep lookup logic explicit; do not silently fall back to the wrong GPU row.

### 6. Write MFU into the metrics dict

- Compute MFU in the engine where step timing and logger access already exist.
- Log MFU with the explicit metric key `perf/mfu`.

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

## Validation

- The current model instance has `calculate_model_flops(...)`, and the method was added by injection instead of by replacing the model class.
- The method signature keeps only parameters required by the actual architecture.
- The engine has a real MFU integration point using runtime step time, device peak FLOPs, precision, and `world_size`.
- The logged MFU value is a `float` under `perf/mfu`.
- The MFU value passes a sanity check: it is not negative, it is not implausibly larger than `1`, and suspicious values were traced back through FLOPs estimation, hardware lookup, precision mapping, timing, and `world_size`.

Add recipe-local assertions under `recipes/<recipe>/skill_tests/model-flops-utilization/asserts.py`,
using the standard `assert_structure(...)` and `assert_smoke(...)` hooks:

- `skill_tests/test_structure.py`: verify recipe structure and MFU wiring.
- `skill_tests/test_smoke.py`: run one real recipe-owned training step and checkpoint/log path.

## Output

- Summarize where `calculate_model_flops(...)` was injected.
- Summarize where engine-side MFU was added or updated.
- State which GPU and precision were used for peak-FLOPs lookup.
- State whether the current run used single-device or multi-device assumptions.
- State where `perf/mfu` is logged.
- If GPU access was unavailable and no local launch instructions existed, state that user input is still required.

## Read On Demand

- Read `references/hardware_peak_flops.csv` when you need peak FLOPs for a known GPU and precision.
- Read the archived sample under `references/recipes/vit_classification_addon/` when you need a concrete MFU integration pattern for config, engine, and model wiring.
