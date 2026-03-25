---
name: model-flops-utilization
description: Implement end-to-end MFU support for the current model and engine, including model FLOPs estimation, hardware peak FLOPs lookup, runtime MFU calculation, and MFU logging. Use when MFU must be added to a real training workflow instead of stopping at FLOPs formulas.
---

# Model FLOPs Utilization

> 中文版：`skills/zh-cn/model/model-flops-utilization/SKILL.md`
> English version: `skills/en/model/model-flops-utilization/SKILL.md`

## Goal

- Add `calculate_model_flops(...) -> float` to the **current model instance** through injection, instead of replacing model classes.
- Add or wire `calculate_mfu(...) -> float` in the engine so MFU is computed from model FLOPs, step time, peak device FLOPs, precision, and `world_size`.
- Log MFU via the existing metrics dictionary using key `perf/mfu`.
- Reuse the demo implementation in `references/recipes/vit_classification_addon/` as a pattern, then adapt to the current recipe.

## Required Inputs

- Model construction entrypoint (where runtime instance is created).
- Engine/training-loop location with step timing and log dict update.
- Real architecture: ViT, decoder-only Transformer, seq2seq encoder-decoder, or VLM (vision + decoder-only text).
- Runtime facts for MFU:
  - precision (`bf16`, `fp16`, `fp32`, ...)
  - `world_size`
  - step time in seconds
  - current device name from `nvidia-smi` or user-provided value

## Workflow

### 1) Confirm environment and hardware

- Detect whether training runs on GPU.
- If yes, get active device name with `nvidia-smi`.
- If CPU-only, scan local `*.md` for cluster/GPU launch instructions; if none are available, ask user how to launch on GPU.

### 2) Inject `calculate_model_flops(...)` into the model instance

- Create `inject_model_flops_calculation(model)` and attach method using `types.MethodType`.
- Keep the method signature minimal and architecture-specific.
- Avoid unused parameters.

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, batch_size: int, seq_len: int, is_training: bool = True) -> float:
        ...

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

Use this in model-building code:

```python
model = AutoModel.from_pretrained(...)
model = inject_model_flops_calculation(model)
```

#### ViT example (no `image_size` / `patch_size` args)

Use model/config attributes (or processor metadata) as the source of token geometry.

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, batch_size: int, is_training: bool = True) -> float:
        batch = int(batch_size)
        if batch <= 0:
            raise ValueError("batch_size must be > 0")

        config = self.config
        hidden = int(config.hidden_size)
        layers = int(config.num_hidden_layers)
        intermediate = int(config.intermediate_size)
        num_labels = int(getattr(config, "num_labels", 1000))

        image_size = int(config.image_size)
        patch_size = int(config.patch_size)
        num_patches = (image_size // patch_size) ** 2
        channels = int(getattr(config, "num_channels", 3))

        patch_embed = 2 * batch * num_patches * (channels * patch_size * patch_size) * hidden
        block_flops = (
            8 * batch * num_patches * hidden * hidden
            + 4 * batch * num_patches * num_patches * hidden
            + 4 * batch * num_patches * hidden * intermediate
        )
        head = 2 * batch * hidden * num_labels
        forward_flops = float(patch_embed + layers * block_flops + head)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### Decoder-only Transformer example

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, batch_size: int, seq_len: int, is_training: bool = True) -> float:
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
        lm_head = 2 * batch * tokens * hidden * vocab
        forward_flops = float(layers * per_layer + lm_head)
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

#### Seq2seq encoder-decoder example (dedicated template)

Use separate source/target lengths and include decoder cross-attention.

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

VLM FLOPs should include both visual and textual stacks.

```python
def calculate_model_flops(...):
    vision_flops = ...
    language_flops = ...
    return float((vision_flops + language_flops) * 3.0 if is_training else (vision_flops + language_flops))
```

### 3) Integrate MFU in engine and recipe logs

- Add/reuse `calculate_mfu(...)`.
- Compute MFU at the same step boundary as the engine timing metric.
- Add MFU to the log dict with key `perf/mfu` instead of direct standalone printing.

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


metrics = {
    "train/loss": float(loss),
    "perf/step_time": float(step_time_seconds),
    "perf/mfu": float(mfu),
}
logger.log(metrics, step=global_step)
```

### 4) Resolve peak FLOPs from hardware table

- Detect GPU via `nvidia-smi`.
- Normalize device name.
- Match precision column in `references/hardware_peak_flops.csv`.
- Ask user when matching is ambiguous.

### 5) Validate end-to-end behavior

- Method exists after injection: `hasattr(model, "calculate_model_flops")`.
- Signature only contains required architecture arguments.
- `calculate_model_flops(...)` and `calculate_mfu(...)` return `float`.
- Engine writes MFU to logs under `perf/mfu`.
- MFU sanity check: non-negative and typically not far above `1`.
- For skill behavior checks, verify both routing trigger and contract check pass.

## Reference demo implementation

Cite these files when explaining or mirroring the integration path:

- `references/recipes/vit_classification_addon/model/vit.py`
- `references/recipes/vit_classification_addon/engine/vit_classification_engine.py`
- `references/recipes/vit_classification_addon/configs/schema.py`
- `references/recipes/vit_classification_addon/configs/train.yaml`

## Output

- Where `inject_model_flops_calculation(model)` is called.
- Where `calculate_mfu(...)` is integrated in the engine.
- Which GPU + precision row was selected from the hardware table.
- Whether MFU assumes single- or multi-GPU (`world_size`).
- Which log dict location carries `perf/mfu`.
