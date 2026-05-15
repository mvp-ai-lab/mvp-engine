---
name: vlm-freeze-policy
description: Understand, modify, or port recipe-local VLM freeze policy for independently configurable trainable groups such as vision encoder, multimodal projector, and language model, including config and optional FLOPs/MFU wiring.
---

# VLM Freeze Policy

## Goal

This skill documents recipe-local VLM freeze policy. Use it in two modes:

- **Basic VLM mode:** If the target recipe is `basic_vlm` or derived from it, do not reimplement freeze policy. Treat this skill as a design map for the existing code and modify only the layer required by the request.
- **Porting mode:** If another VLM recipe does not have independent freeze controls, use this skill to add them by adapting the `basic_vlm` design to the target model's real parameter names.

Keep the implementation recipe-local unless the user explicitly asks for engine-wide behavior.

## Existing Reference Implementation

Reference in `basic_vlm`:

- `recipes/basic_vlm/configs/schema.py` defines `freeze_vit`, `freeze_merger`, and `freeze_llm`.
- `recipes/basic_vlm/configs/stage*.yaml` shows stage-specific freeze choices.
- `recipes/basic_vlm/model/qwen3_vl.py` owns group prefixes, `apply_freeze_policy`, build order, and freeze-aware FLOPs.
- `recipes/basic_vlm/engine/basic_vlm_engine.py` filters trainable parameters and passes freeze flags into FLOPs calculation.
- `recipes/basic_vlm/utils/log/mfu.py` consumes precomputed model FLOPs for MFU logging.

## Workflow

### 1. Identify The Starting Point

Search the target recipe:

```bash
rg -n "freeze_|requires_grad|apply_freeze_policy|trainable|calculate_model_flops|mfu" recipes/<recipe> mvp_engine
```

- If the recipe already follows `basic_vlm`, read the relevant existing layer and make a local change.
- If the recipe has no freeze policy, use `recipes/basic_vlm/model/qwen3_vl.py` as the implementation reference, but derive prefixes from the target model.
- Do not assume every VLM uses Qwen3-VL names such as `visual`, `merger`, or `language_model`.

### 2. Parameter Groups

Typical VLM groups are:

- vision encoder: patch embedding, vision tower, or ViT blocks;
- connector: projector, merger, adapter, resampler, or cross-modal bridge;
- language stack: text backbone and output head.

Guidelines:

- Use `model.named_parameters()` from the real model.
- Keep groups non-overlapping and deterministic.
- Prefer prefixes or module-path predicates over fragile substring checks.
- Keep connector/projector separate from vision so alignment stages can train only the connector.
- Keep output heads with the language stack unless the recipe has a concrete reason to control them separately.

Reference in `basic_vlm`:

- `recipes/basic_vlm/model/qwen3_vl.py` defines `VIT_PREFIXES`, `MERGER_PREFIXES`, `LLM_PREFIXES`, `_matches`, and `apply_freeze_policy`.

### 3. Build Order

Call the freeze helper after the model has all trainable modules attached or patched, and before anything consumes `requires_grad`.

Common ordering:

1. load model;
2. apply model compatibility patches, forward injections, and checkpointing hooks;
3. apply freeze policy;
4. upcast or count trainable parameters;
5. parallelize and build optimizer.

Reference in `basic_vlm`:

- `recipes/basic_vlm/model/qwen3_vl.py` applies freeze policy inside `build_qwen3_vl_model`.
- `recipes/basic_vlm/engine/basic_vlm_engine.py` builds the optimizer from parameters where `requires_grad` is true.

### 4. Config And Stages

Add freeze fields near related model-loading or training-stage options. Choose defaults from the recipe's intended default training stage, not from a hard-coded global rule.

Example shape:

```yaml
model:
  freeze_vit: true
  freeze_merger: false
  freeze_llm: true
```

Guidelines:

- Preserve existing names when the recipe already has conventions such as `freeze_vit`, `freeze_merger`, or `freeze_llm`.
- Make stage YAMLs explicit when stages train different components.
- Do not add disabled fields to unrelated YAMLs only to restate schema defaults.

Reference in `basic_vlm`:

- `recipes/basic_vlm/configs/stage1.yaml` trains the merger while freezing ViT and LLM.
- `recipes/basic_vlm/configs/stage2.yaml` and `stage3.yaml` unfreeze all three groups.

### 5. Optimizer, Metrics, And MFU

Check every path that depends on trainability:

- optimizer parameter collection;
- trainable parameter counts or logs;
- fp32 trainable-parameter upcasting;
- distributed wrapping assumptions;
- FLOPs/MFU estimation.

If the recipe estimates training FLOPs by component, make the estimate freeze-aware. For a serial VLM path, reason backward from the loss: language stack, connector, then vision encoder. A frozen component may still need activation-gradient cost when trainable upstream modules depend on it.

Reference in `basic_vlm`:

- `recipes/basic_vlm/model/qwen3_vl.py` computes freeze-aware FLOPs in `inject_model_flops_calculation`.
- `recipes/basic_vlm/engine/basic_vlm_engine.py` passes freeze flags when collecting `model_flops`.
- `recipes/basic_vlm/utils/log/mfu.py` logs MFU from the precomputed per-step FLOPs.

## Validation

- Confirm no stale broad freeze overrides independent group controls.
- Confirm every config/YAML freeze flag is consumed by the model builder.
- Confirm freezing runs before optimizer construction and parallel wrapping.
- Confirm the intended stage has at least one trainable parameter.
- If FLOPs/MFU exists, confirm freeze flags affect the FLOPs path only where the estimate is computed.

## Output

- State whether you modified existing `basic_vlm` behavior or ported freeze policy into another VLM recipe.
- State the groups, config flags, build-order location, and optimizer/MFU impact.
- State what validation ran and what remains unverified.
