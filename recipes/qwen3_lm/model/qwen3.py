"""Qwen3 model helpers for the Qwen3 LM recipe."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        MoeCausalLMOutputWithPast,
    )
except ImportError:  # pragma: no cover - depends on the installed transformers version.
    MoeCausalLMOutputWithPast = None


def _slice_hidden(hidden_states: torch.Tensor, logits_to_keep: int | torch.Tensor) -> torch.Tensor:
    """Slice trailing hidden states based on ``logits_to_keep``."""
    sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    if hidden_states.dim() == 3:
        return hidden_states[:, sl, :]
    return hidden_states[sl, :]


def _shift_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Pad-right by one token then shift left for causal LM loss."""
    labels = F.pad(labels, (0, 1), value=ignore_index)
    return labels[..., 1:].contiguous()


def apply_model_gradient_checkpointing(
    model,
    *,
    enabled: bool = False,
    use_reentrant: bool = False,
):
    """Enable the model's built-in gradient checkpointing when requested by the recipe."""
    if not enabled:
        return model

    if not hasattr(model, "gradient_checkpointing_enable"):
        raise AttributeError(f"{model.__class__.__name__} does not support gradient checkpointing.")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    setattr(model.config, "use_cache", False)
    return model


def upcast_trainable_params_to_fp32(model):
    """Storing trainable weights in fp32 master precision."""
    for _, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(dtype=torch.float32)
    return model


def inject_model_flops_calculation(model):
    """Inject per-process FLOPs estimation onto the loaded Qwen3 model."""

    def _count_sparse_layers(cfg: Any) -> int:
        """Count sparse MoE layers in a Qwen3-MoE config."""
        num_experts = int(getattr(cfg, "num_experts", 0) or 0)
        if num_experts <= 0:
            return 0

        sparse_step = int(getattr(cfg, "decoder_sparse_step", 1) or 1)
        mlp_only_layers = set(int(layer) for layer in getattr(cfg, "mlp_only_layers", []) or [])
        return sum(
            1
            for layer_idx in range(int(cfg.num_hidden_layers))
            if layer_idx not in mlp_only_layers and (layer_idx + 1) % sparse_step == 0
        )

    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        seq_len: int,
        attention_mask: torch.Tensor | None = None,
        is_training: bool = True,
    ) -> float:
        """Estimate local-rank Qwen3 FLOPs for one batch."""
        batch = int(batch_size)
        tokens = int(seq_len)
        if batch <= 0 or tokens <= 0:
            raise ValueError("batch_size and seq_len must be > 0")

        cfg = self.config
        layers = int(cfg.num_hidden_layers)
        hidden = int(cfg.hidden_size)
        vocab = int(cfg.vocab_size)
        head_dim = int(getattr(cfg, "head_dim", hidden // int(cfg.num_attention_heads)))
        query_hidden = int(cfg.num_attention_heads) * head_dim
        kv_hidden = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)) * head_dim
        attention_token_pairs = batch * tokens * tokens
        if attention_mask is not None and attention_mask.ndim == 2:
            mask = attention_mask.detach().to(device="cpu", dtype=torch.long)
            max_mask_value = int(mask.max().item()) if mask.numel() > 0 else 0
            if max_mask_value > 1:
                attention_token_pairs = 0
                for row in mask:
                    segment_ids = row[row > 0]
                    if segment_ids.numel() == 0:
                        continue
                    segment_lengths = torch.bincount(segment_ids, minlength=max_mask_value + 1)[1:]
                    attention_token_pairs += int(torch.square(segment_lengths).sum().item())
            else:
                valid_lengths = mask.ne(0).sum(dim=-1)
                attention_token_pairs = int(torch.square(valid_lengths).sum().item())

            if attention_token_pairs <= 0:
                raise ValueError("attention_mask must contain at least one valid token")

        token_count = batch * tokens
        attention_per_layer = (
            2 * token_count * hidden * (query_hidden + 2 * kv_hidden)
            + 4 * attention_token_pairs * query_hidden
            + 2 * token_count * query_hidden * hidden
        )

        sparse_layers = _count_sparse_layers(cfg)
        dense_layers = layers - sparse_layers
        dense_mlp_flops = dense_layers * 6 * token_count * hidden * int(cfg.intermediate_size)
        sparse_mlp_flops = 0
        if sparse_layers:
            sparse_mlp_flops = sparse_layers * (
                2 * token_count * hidden * int(cfg.num_experts)
                + 6 * token_count * hidden * int(cfg.moe_intermediate_size) * int(cfg.num_experts_per_tok)
            )

        language_flops = float(
            layers * attention_per_layer + dense_mlp_flops + sparse_mlp_flops + 2 * token_count * hidden * vocab
        )
        return language_flops * (3.0 if is_training else 1.0)

    model.calculate_model_flops = MethodType(calculate_model_flops, model)
    return model


def inject_sum_loss_forward(model, *, chunk_size: int = 4096):
    """Patch Qwen3 forward to return unreduced per-token CE loss."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_router_logits=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        """Forward Qwen3 with chunked per-token CE instead of mean loss."""
        if hasattr(self.config, "output_router_logits"):
            output_router_logits = (
                output_router_logits if output_router_logits is not None else self.config.output_router_logits
            )
            if labels is not None and output_router_logits:
                raise NotImplementedError(
                    "Qwen3 LM token-normalized MoE training does not support router auxiliary loss yet. "
                    "Set `model.config.output_router_logits=false` or extend the engine to normalize aux loss."
                )

        model_kwargs = dict(kwargs)
        if hasattr(self.config, "output_router_logits"):
            model_kwargs["output_router_logits"] = output_router_logits

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **model_kwargs,
        )
        hidden_states = getattr(outputs, "last_hidden_state", outputs[0])

        loss = None
        logits = None
        if labels is not None:
            hs = _slice_hidden(hidden_states, logits_to_keep)
            shift_labels = _shift_labels(labels)
            if hs.dim() == 3 and hs.shape[1] != shift_labels.shape[-1]:
                shift_labels = shift_labels[..., -hs.shape[1] :]
            flat_hs = hs.reshape(-1, hs.size(-1))
            flat_labels = shift_labels.reshape(-1)
            lm_head_bias = getattr(self.lm_head, "bias", None)

            loss_chunks: list[torch.Tensor] = []
            for start in range(0, flat_hs.size(0), chunk_size):
                end = min(start + chunk_size, flat_hs.size(0))
                chunk_logits = F.linear(flat_hs[start:end], self.lm_head.weight, lm_head_bias)
                loss_chunks.append(
                    F.cross_entropy(
                        chunk_logits,
                        flat_labels[start:end],
                        ignore_index=-100,
                        reduction="none",
                    )
                )
                del chunk_logits
            loss = torch.cat(loss_chunks, dim=0)
        else:
            logits = self.lm_head(_slice_hidden(hidden_states, logits_to_keep))

        output_payload = {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
        if hasattr(outputs, "router_logits") and MoeCausalLMOutputWithPast is not None:
            return MoeCausalLMOutputWithPast(
                **output_payload,
                aux_loss=None,
                router_logits=outputs.router_logits,
            )

        return CausalLMOutputWithPast(
            **output_payload,
        )

    model.forward = MethodType(forward, model)
    return model


def build_tiny_qwen3_model(model_config: Any):
    """Build a tiny random Qwen3 model for offline smoke tests."""
    hidden_size = int(model_config.tiny_hidden_size)
    num_attention_heads = int(model_config.tiny_num_attention_heads)
    config = Qwen3Config(
        vocab_size=int(model_config.tiny_vocab_size),
        hidden_size=hidden_size,
        intermediate_size=int(model_config.tiny_intermediate_size),
        num_hidden_layers=int(model_config.tiny_num_hidden_layers),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(model_config.tiny_num_key_value_heads),
        head_dim=hidden_size // num_attention_heads,
        max_position_embeddings=2048,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
    )
    model = Qwen3ForCausalLM(config)
    model.config._attn_implementation = model_config.attn_implementation
    return model


def build_qwen3_model(model_config: Any):
    """Load or build the Qwen3 model and apply recipe-local runtime patches."""
    if bool(model_config.tiny_random):
        model = build_tiny_qwen3_model(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.pretrained_model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation=model_config.attn_implementation,
        )

    model = inject_model_flops_calculation(model)
    model = apply_model_gradient_checkpointing(
        model,
        enabled=bool(model_config.gradient_checkpointing.enabled),
        use_reentrant=bool(model_config.gradient_checkpointing.use_reentrant),
    )
    model = inject_sum_loss_forward(model, chunk_size=int(model_config.loss_chunk_size))
    if bool(model_config.upcast_trainable_params):
        model = upcast_trainable_params_to_fp32(model)
    return model
