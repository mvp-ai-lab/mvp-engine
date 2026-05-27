"""Reusable token-normalized loss accounting."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class TokenLossStats:
    """Reduced accumulation-window token loss statistics."""

    global_total_tokens: int
    global_effective_tokens: int
    global_loss_sum: float
    backward_divisor: int
    gradient_scale: float


class TokenNormedLossKit:
    """Track one token-normalized gradient-accumulation window."""

    def __init__(
        self,
        *,
        device: torch.device,
        dp_world_size: int,
        dp_group: dist.ProcessGroup | None = None,
    ) -> None:
        """Create an empty token-loss window on the given reduction device."""
        self.device = device
        self.dp_world_size = int(dp_world_size)
        self.dp_group = dp_group
        self.reset()

    def apply_chunked_token_loss_patch(
        self,
        model: torch.nn.Module,
        *,
        chunk_size: int = 4096,
        output_cls: type | None = None,
    ) -> torch.nn.Module:
        """Patch a causal LM to return unreduced per-token loss with chunked logits."""
        return apply_chunked_token_loss_patch(
            model,
            chunk_size=chunk_size,
            output_cls=output_cls,
        )

    def accumulate_microbatch(
        self,
        *,
        loss_sum: torch.Tensor,
        effective_tokens: int,
        total_tokens: int,
        backward_divisor: int,
    ) -> torch.Tensor:
        """Accumulate one micro-batch and return its provisional backward loss."""
        if backward_divisor <= 0:
            raise ValueError("backward_divisor must be positive.")
        if effective_tokens < 0 or total_tokens < 0:
            raise ValueError("Token counts must be non-negative.")
        if self._backward_divisor is None:
            self._backward_divisor = int(backward_divisor)
        elif self._backward_divisor != int(backward_divisor):
            raise ValueError("backward_divisor must stay fixed within one accumulation window.")

        self._total_tokens += int(total_tokens)
        self._effective_tokens += int(effective_tokens)
        detached_loss = loss_sum.detach().to(device=self.device, dtype=torch.float64)
        self._loss_sum = detached_loss if self._loss_sum is None else self._loss_sum + detached_loss
        return loss_sum / float(backward_divisor)

    def reduce_window(self) -> TokenLossStats:
        """Reduce local window stats across distributed ranks."""
        if self._backward_divisor is None or self._loss_sum is None:
            raise RuntimeError("No token loss has been accumulated.")

        token_values = torch.tensor(
            [self._total_tokens, self._effective_tokens],
            device=self.device,
            dtype=torch.float64,
        )
        loss_sum = self._loss_sum.clone()
        if dist.is_available() and dist.is_initialized() and self.dp_world_size > 1:
            dist.all_reduce(token_values, op=dist.ReduceOp.SUM, group=self.dp_group)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=self.dp_group)

        global_effective_tokens = int(token_values[1].item())
        if global_effective_tokens <= 0:
            raise ValueError("Accumulation window must contain at least one supervised token.")

        gradient_scale = float(self._backward_divisor) * float(self.dp_world_size) / float(global_effective_tokens)
        return TokenLossStats(
            global_total_tokens=int(token_values[0].item()),
            global_effective_tokens=global_effective_tokens,
            global_loss_sum=float(loss_sum.item()),
            backward_divisor=int(self._backward_divisor),
            gradient_scale=gradient_scale,
        )

    def rescale_gradients(self, parameters: Iterable[torch.nn.Parameter], stats: TokenLossStats) -> None:
        """Apply the final token-normalization factor to accumulated gradients."""
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(stats.gradient_scale)

    def reset(self) -> None:
        """Clear all state for a new accumulation window."""
        self._total_tokens = 0
        self._effective_tokens = 0
        self._loss_sum: torch.Tensor | None = None
        self._backward_divisor: int | None = None


def apply_chunked_token_loss_patch(
    model: torch.nn.Module,
    *,
    chunk_size: int = 4096,
    output_cls: type | None = None,
) -> torch.nn.Module:
    """Patch a causal LM forward to compute unreduced CE loss in logits chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not hasattr(model, "model"):
        raise ValueError("Chunked token loss patch requires the model to expose `.model`.")
    if not hasattr(model, "lm_head"):
        raise ValueError("Chunked token loss patch requires the model to expose `.lm_head`.")

    if output_cls is None:
        from transformers.modeling_outputs import CausalLMOutputWithPast

        output_cls = CausalLMOutputWithPast

    original_forward = model.forward
    outer_sig = inspect.signature(original_forward)
    inner_sig = inspect.signature(model.model.forward)
    inner_accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in inner_sig.parameters.values())
    inner_param_names = set(inner_sig.parameters)
    output_sig = inspect.signature(output_cls)
    output_param_names = set(output_sig.parameters)
    lm_head_bias = getattr(model.lm_head, "bias", None)

    def forward(self, *args: Any, **kwargs: Any):
        if args:
            bound = outer_sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            forward_kwargs = dict(bound.arguments)
            extra_kwargs = forward_kwargs.pop("kwargs", {})
            forward_kwargs.update(extra_kwargs)
        else:
            forward_kwargs = dict(kwargs)

        labels = forward_kwargs.pop("labels", None)
        logits_to_keep = forward_kwargs.pop("logits_to_keep", 0)
        if not inner_accepts_kwargs:
            forward_kwargs = {name: value for name, value in forward_kwargs.items() if name in inner_param_names}

        outputs = self.model(**forward_kwargs)
        hidden_states = outputs[0]
        logits_slice = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = (
            hidden_states[:, logits_slice, :] if hidden_states.dim() == 3 else hidden_states[logits_slice, :]
        )

        loss = None
        logits = None
        if labels is None:
            logits = self.lm_head(hidden_states)
        else:
            shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            shift_labels = shift_labels[:, logits_slice] if shift_labels.dim() == 2 else shift_labels[logits_slice]
            flat_hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            flat_labels = shift_labels.reshape(-1)
            loss_function = getattr(self, "loss_function", None)
            loss_chunks = []
            for start in range(0, flat_hidden_states.size(0), chunk_size):
                end = min(start + chunk_size, flat_hidden_states.size(0))
                chunk_logits = F.linear(flat_hidden_states[start:end], self.lm_head.weight, lm_head_bias)
                chunk_labels = flat_labels[start:end]
                chunk_loss = None
                if loss_function is not None:
                    chunk_loss = loss_function(
                        logits=chunk_logits,
                        labels=chunk_labels,
                        vocab_size=chunk_logits.size(-1),
                        shift_labels=chunk_labels,
                    )
                if chunk_loss is None or chunk_loss.ndim == 0:
                    chunk_loss = F.cross_entropy(
                        chunk_logits,
                        chunk_labels,
                        ignore_index=-100,
                        reduction="none",
                    )
                loss_chunks.append(chunk_loss.reshape(-1))
            loss = torch.cat(loss_chunks, dim=0)

        output_kwargs = {
            "loss": loss,
            "logits": logits,
            "past_key_values": getattr(outputs, "past_key_values", None),
            "hidden_states": getattr(outputs, "hidden_states", None),
            "attentions": getattr(outputs, "attentions", None),
            "rope_deltas": getattr(outputs, "rope_deltas", None),
        }
        output_kwargs = {name: value for name, value in output_kwargs.items() if name in output_param_names}
        return output_cls(**output_kwargs)

    forward.__signature__ = outer_sig
    model.forward = MethodType(forward, model)
    return model
