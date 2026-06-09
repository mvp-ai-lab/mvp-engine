"""Qwen3-VL model helpers for the minimal VLM recipe."""

from __future__ import annotations

import weakref
from types import MethodType
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoModelForImageTextToText

from mvp_engine.distributed.cp import build_long_context_attention
from mvp_engine.distributed.utils import (
    MESH_DIM_TENSOR,
    get_context_parallel_group,
    get_context_parallel_size,
    get_mesh_dim_size,
    get_tensor_parallel_mesh,
)
from mvp_engine.utils.log import logger

MERGER_PREFIXES = (
    "model.visual.merger.",
    "model.visual.deepstack_merger_list.",
)

QWEN3_VL_TP_MODULE_CONFIG: dict[str, dict[str, str]] = {
    "Qwen3VLTextAttention": {
        "q_proj": "col",
        "k_proj": "col",
        "v_proj": "col",
        "o_proj": "row",
    },
    "Qwen3VLTextMLP": {
        "gate_proj": "col",
        "up_proj": "col",
        "down_proj": "row",
    },
}

QWEN3_VL_SEQUENCE_PARALLEL_MODULE_CONFIG: dict[str, dict[str, str]] = {
    "Qwen3VLTextModel": {
        "norm": "sequence",
    },
    "Qwen3VLTextDecoderLayer": {
        "input_layernorm": "sequence",
        "post_attention_layernorm": "sequence",
    },
}

_SP_BOUNDARY_CONFIGURED_ATTR = "_qwen3_vl_sequence_parallel_boundary_configured"
_SP_BOUNDARY_HANDLES_ATTR = "_qwen3_vl_sequence_parallel_boundary_handles"
_TP_QK_NORM_GRAD_SYNC_ATTR = "_qwen3_vl_tp_qk_norm_grad_sync_configured"
_TP_QK_NORM_GRAD_SYNC_HANDLES_ATTR = "_qwen3_vl_tp_qk_norm_grad_sync_handles"


def apply_freeze_policy(model) -> int:
    """Freeze the visual encoder and merger for the default demo setup.

    Args:
        model: Loaded Qwen3-VL model instance.

    Returns:
        The number of parameters that were marked non-trainable.
    """
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        if name.startswith("model.visual.") or any(name.startswith(prefix) for prefix in MERGER_PREFIXES):
            parameter.requires_grad = False
            frozen_parameters += parameter.numel()

    return frozen_parameters


def prepare_qwen3_vl_mrope_position_ids(model: torch.nn.Module, batch: dict[str, Any]) -> None:
    """Populate Qwen3-VL multimodal RoPE position ids for full, unsharded batches."""
    if "position_ids" in batch:
        return

    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return

    has_image = isinstance(batch.get("image_grid_thw"), torch.Tensor)
    has_video = isinstance(batch.get("video_grid_thw"), torch.Tensor)
    if not has_image and not has_video:
        return

    qwen_model = _resolve_qwen3_vl_base_model(model)
    mm_token_type_ids = batch.get("mm_token_type_ids")
    if not isinstance(mm_token_type_ids, torch.Tensor):
        mm_token_type_ids = _infer_qwen3_vl_mm_token_type_ids(
            model,
            input_ids,
            has_image=has_image,
            has_video=has_video,
        )
        batch["mm_token_type_ids"] = mm_token_type_ids

    position_ids, rope_deltas = qwen_model.get_rope_index(
        input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        attention_mask=batch.get("attention_mask"),
    )
    batch["position_ids"] = position_ids
    qwen_model.rope_deltas = rope_deltas


def _resolve_qwen3_vl_base_model(model: torch.nn.Module) -> torch.nn.Module:
    qwen_model = getattr(model, "model", model)
    if hasattr(qwen_model, "get_rope_index"):
        return qwen_model
    raise TypeError(f"Expected Qwen3VL model with get_rope_index, got {model.__class__.__name__}.")


def _infer_qwen3_vl_mm_token_type_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    has_image: bool,
    has_video: bool,
) -> torch.Tensor:
    config = getattr(model, "config", None)
    token_type_ids = torch.zeros_like(input_ids)

    if has_image:
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("Cannot infer Qwen3-VL image token types without config.image_token_id.")
        token_type_ids = token_type_ids.masked_fill(input_ids == int(image_token_id), 1)

    if has_video:
        video_token_id = getattr(config, "video_token_id", None)
        if video_token_id is None:
            raise ValueError("Cannot infer Qwen3-VL video token types without config.video_token_id.")
        token_type_ids = token_type_ids.masked_fill(input_ids == int(video_token_id), 2)

    return token_type_ids


def _get_autocast_attention_dtype(x: torch.Tensor) -> torch.dtype | None:
    if not x.is_floating_point() or not torch.is_autocast_enabled(x.device.type):
        return None
    dtype = torch.get_autocast_dtype(x.device.type)
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return None


def _forward_qwen3_vl_text_attention_long_context(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run Qwen3-VL text attention over context-local sequence shards."""
    if past_key_values is not None:
        raise ValueError("Minimal VLM long-context attention does not support KV-cache training inputs.")
    _ = attention_mask

    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    query_projected = self.q_proj(hidden_states)
    key_projected = self.k_proj(hidden_states)
    value_projected = self.v_proj(hidden_states)

    input_shape = query_projected.shape[:-1]
    query_shape = (*input_shape, -1, self.head_dim)
    key_shape = (*key_projected.shape[:-1], -1, self.head_dim)
    value_shape = (*value_projected.shape[:-1], -1, self.head_dim)

    query_states = self.q_norm(query_projected.view(query_shape)).transpose(1, 2)
    key_states = self.k_norm(key_projected.view(key_shape)).transpose(1, 2)
    value_states = value_projected.view(value_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    attn_dtype = _get_autocast_attention_dtype(hidden_states)
    if attn_dtype is not None:
        query_states = query_states.to(attn_dtype)
        key_states = key_states.to(attn_dtype)
        value_states = value_states.to(attn_dtype)

    attn_output = self.long_context_attention(
        query_states,
        key_states,
        value_states,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        softmax_scale=self.scaling,
        causal=True,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _forward_qwen3_vl_text_attention_tensor_parallel(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3-VL text attention after TP/SP projection layout changes."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    query_projected = self.q_proj(hidden_states)
    key_projected = self.k_proj(hidden_states)
    value_projected = self.v_proj(hidden_states)

    input_shape = query_projected.shape[:-1]
    query_shape = (*input_shape, -1, self.head_dim)
    key_shape = (*key_projected.shape[:-1], -1, self.head_dim)
    value_shape = (*value_projected.shape[:-1], -1, self.head_dim)

    query_states = self.q_norm(query_projected.view(query_shape)).transpose(1, 2)
    key_states = self.k_norm(key_projected.view(key_shape)).transpose(1, 2)
    value_states = value_projected.view(value_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _postprocess_qwen3_vl_text_attention_for_tp(module: torch.nn.Module, tp_mesh: DeviceMesh) -> None:
    if int(tp_mesh.size()) <= 1 or getattr(module, "_qwen3_vl_tp_attention_forward_configured", False):
        return
    _validate_qwen3_vl_attention_tensor_parallel(module, int(tp_mesh.size()))
    if getattr(module, "_long_context_attention_configured", False):
        return

    module._qwen3_vl_original_forward = module.forward
    module.forward = MethodType(_forward_qwen3_vl_text_attention_tensor_parallel, module)
    module._qwen3_vl_tp_attention_forward_configured = True


def _postprocess_qwen3_vl_text_model_for_sequence_parallel(module: torch.nn.Module, tp_mesh: DeviceMesh) -> None:
    if int(tp_mesh.size()) <= 1 or getattr(module, _SP_BOUNDARY_CONFIGURED_ATTR, False):
        return

    layers = getattr(module, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Qwen3-VL sequence parallel requires at least one text decoder layer.")

    handles = [
        layers[0].register_forward_pre_hook(
            _make_sequence_parallel_input_shard_hook(tp_mesh),
            with_kwargs=True,
        ),
        module.register_forward_hook(_make_sequence_parallel_output_gather_hook(tp_mesh)),
    ]
    setattr(module, _SP_BOUNDARY_HANDLES_ATTR, handles)
    setattr(module, _SP_BOUNDARY_CONFIGURED_ATTR, True)


def _make_sequence_parallel_input_shard_hook(tp_mesh: DeviceMesh):
    def hook(_module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if args:
            hidden_states = args[0]
            if isinstance(hidden_states, torch.Tensor):
                args = (
                    _redistribute_sequence_tensor(hidden_states, tp_mesh, source="replicate", target="shard"),
                    *args[1:],
                )
        else:
            hidden_states = kwargs.get("hidden_states")
            if isinstance(hidden_states, torch.Tensor):
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = _redistribute_sequence_tensor(
                    hidden_states,
                    tp_mesh,
                    source="replicate",
                    target="shard",
                )
        return args, kwargs

    return hook


def _make_sequence_parallel_output_gather_hook(tp_mesh: DeviceMesh):
    def hook(_module, _args: tuple[Any, ...], output: Any) -> Any:
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if isinstance(output, BaseModelOutputWithPast):
            return BaseModelOutputWithPast(
                last_hidden_state=_redistribute_sequence_tensor(
                    output.last_hidden_state,
                    tp_mesh,
                    source="shard",
                    target="replicate",
                ),
                past_key_values=output.past_key_values,
                hidden_states=_redistribute_sequence_tensor_tree(
                    output.hidden_states,
                    tp_mesh,
                    source="shard",
                    target="replicate",
                ),
                attentions=output.attentions,
            )
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (
                _redistribute_sequence_tensor(output[0], tp_mesh, source="shard", target="replicate"),
                *output[1:],
            )
        if isinstance(output, torch.Tensor):
            return _redistribute_sequence_tensor(output, tp_mesh, source="shard", target="replicate")
        return output

    return hook


def _redistribute_sequence_tensor_tree(
    value: Any,
    tp_mesh: DeviceMesh,
    *,
    source: str,
    target: str,
) -> Any:
    if isinstance(value, torch.Tensor):
        return _redistribute_sequence_tensor(value, tp_mesh, source=source, target=target)
    if isinstance(value, tuple):
        return tuple(_redistribute_sequence_tensor_tree(item, tp_mesh, source=source, target=target) for item in value)
    return value


def _redistribute_sequence_tensor(
    tensor: torch.Tensor,
    tp_mesh: DeviceMesh,
    *,
    source: str,
    target: str,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor, Replicate, Shard

    source_layout = Replicate() if source == "replicate" else Shard(1)
    target_layout = Replicate() if target == "replicate" else Shard(1)
    if isinstance(tensor, DTensor):
        distributed = tensor
    else:
        distributed = DTensor.from_local(tensor, tp_mesh, (source_layout,), run_check=False)
    if distributed.placements != (target_layout,):
        distributed = distributed.redistribute(placements=(target_layout,), async_op=True)
    local_tensor = distributed.to_local()
    wait = getattr(local_tensor, "wait", None)
    if callable(wait):
        local_tensor = wait()
    return local_tensor


QWEN3_VL_TP_MODULE_POSTPROCESSORS = {
    "Qwen3VLTextAttention": _postprocess_qwen3_vl_text_attention_for_tp,
    "Qwen3VLTextModel": _postprocess_qwen3_vl_text_model_for_sequence_parallel,
}


def _configure_text_attention_long_context(
    module: torch.nn.Module,
    config: Mapping[str, Any],
    device_mesh: DeviceMesh,
) -> None:
    """Patch one Qwen3-VL text attention module for Ulysses long-context attention."""
    if getattr(module, "_long_context_attention_configured", False):
        return

    context_size = get_context_parallel_size(device_mesh)
    text_config = getattr(module, "config", None)
    if text_config is not None:
        tensor_size = get_mesh_dim_size(device_mesh, MESH_DIM_TENSOR)
        num_heads = int(text_config.num_attention_heads)
        num_key_value_heads = int(text_config.num_key_value_heads)
        if num_heads % tensor_size != 0 or num_key_value_heads % tensor_size != 0:
            raise ValueError(
                "Qwen3-VL parallel.mesh.tensor must divide both num_attention_heads and num_key_value_heads."
            )
        local_num_heads = num_heads // tensor_size
        local_num_key_value_heads = num_key_value_heads // tensor_size
        if local_num_heads % context_size != 0 or local_num_key_value_heads % context_size != 0:
            raise ValueError("Qwen3-VL parallel.mesh.context must divide TP-local attention and key-value heads.")

    module.long_context_attention = build_long_context_attention(config, device_mesh)
    module._long_context_original_forward = module.forward
    module.forward = MethodType(_forward_qwen3_vl_text_attention_long_context, module)
    module._long_context_attention_configured = True


def apply_long_context_attention_for_qwen3_vl(
    model: torch.nn.Module,
    device_mesh: DeviceMesh,
    long_context_config: Mapping[str, Any],
) -> None:
    """Install long-context attention on Qwen3-VL text attention modules."""
    if getattr(model, "_long_context_attention_configured", False):
        return

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    model._long_context_device_mesh = device_mesh
    model._long_context_config = dict(long_context_config)
    configured_count = 0
    for module in model.modules():
        if isinstance(module, Qwen3VLTextAttention):
            _configure_text_attention_long_context(module, long_context_config, device_mesh)
            configured_count += 1

    if configured_count == 0:
        raise ValueError("Qwen3-VL long-context hook did not find any Qwen3VLTextAttention modules.")
    model._long_context_attention_configured = True


def install_qwen3_vl_tensor_parallel_grad_sync(model: torch.nn.Module, device_mesh: DeviceMesh) -> None:
    """Synchronize replicated Q/K norm gradients across the tensor-parallel mesh."""
    tensor_size = get_mesh_dim_size(device_mesh, MESH_DIM_TENSOR)
    if tensor_size <= 1:
        return
    if getattr(model, _TP_QK_NORM_GRAD_SYNC_ATTR, False):
        return
    if not dist.is_available() or not dist.is_initialized():
        return

    tp_group = get_tensor_parallel_mesh(device_mesh).get_group()
    qk_norm_parameters = _collect_qwen3_vl_qk_norm_parameters(model, tensor_size)
    qk_norm_parameter_ids = {id(parameter) for parameter in qk_norm_parameters}
    context_size = get_context_parallel_size(device_mesh)
    context_sync_configured = bool(getattr(model, "_long_context_grad_sync_configured", False))

    if context_size > 1 and context_sync_configured:
        handles = _replace_context_grad_sync_for_tensor_parallel_qk_norms(
            model,
            qk_norm_parameter_ids=qk_norm_parameter_ids,
            tp_group=tp_group,
            context_group=get_context_parallel_group(device_mesh),
        )
        qk_handles = [handle for parameter_id, handle in handles if parameter_id in qk_norm_parameter_ids]
        if len(qk_handles) != len(qk_norm_parameters):
            raise ValueError("Qwen3-VL tensor-parallel Q/K norm grad sync missed parameters after FSDP wrapping.")
        model._long_context_grad_sync_handles = [handle for _, handle in handles]
        logger.info(
            "Reinstalled long-context grad sync with tensor-parallel Q/K norm sync "
            f"on {len(qk_handles)} replicated parameters."
        )
    else:
        qk_handles = [
            parameter.register_post_accumulate_grad_hook(_make_grad_sync_hook((tp_group,)))
            for parameter in qk_norm_parameters
        ]
        logger.info(f"Installed tensor-parallel Q/K norm grad sync hooks on {len(qk_handles)} parameters.")

    setattr(model, _TP_QK_NORM_GRAD_SYNC_HANDLES_ATTR, qk_handles)
    setattr(model, _TP_QK_NORM_GRAD_SYNC_ATTR, True)


def _collect_qwen3_vl_qk_norm_parameters(
    model: torch.nn.Module,
    tensor_size: int,
) -> list[torch.nn.Parameter]:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    parameters: list[torch.nn.Parameter] = []
    attention_count = 0
    seen_parameter_ids: set[int] = set()
    for module in model.modules():
        if not isinstance(module, Qwen3VLTextAttention):
            continue

        _validate_qwen3_vl_attention_tensor_parallel(module, tensor_size)
        attention_count += 1
        for norm_name in ("q_norm", "k_norm"):
            norm = getattr(module, norm_name, None)
            weight = getattr(norm, "weight", None)
            if weight is None or not weight.requires_grad or id(weight) in seen_parameter_ids:
                continue
            parameters.append(weight)
            seen_parameter_ids.add(id(weight))

    if attention_count == 0:
        raise ValueError("Qwen3-VL tensor-parallel grad sync did not find any text attention modules.")
    if not parameters:
        raise ValueError("Qwen3-VL tensor-parallel grad sync did not find trainable Q/K norm weights.")
    return parameters


def _validate_qwen3_vl_attention_tensor_parallel(module: torch.nn.Module, tensor_size: int) -> None:
    text_config = getattr(module, "config", None)
    if text_config is None:
        return

    num_heads = int(text_config.num_attention_heads)
    num_key_value_heads = int(text_config.num_key_value_heads)
    if num_heads % tensor_size != 0 or num_key_value_heads % tensor_size != 0:
        raise ValueError("Qwen3-VL tensor parallel size must divide both num_attention_heads and num_key_value_heads.")


def _replace_context_grad_sync_for_tensor_parallel_qk_norms(
    model: torch.nn.Module,
    *,
    qk_norm_parameter_ids: set[int],
    tp_group: dist.ProcessGroup,
    context_group: dist.ProcessGroup | None,
) -> list[tuple[int, torch.utils.hooks.RemovableHandle]]:
    if context_group is None:
        raise ValueError("Qwen3-VL context+tensor grad sync requires an active context process group.")

    for handle in getattr(model, "_long_context_grad_sync_handles", []):
        handle.remove()

    handles: list[tuple[int, torch.utils.hooks.RemovableHandle]] = []
    seen_parameter_ids: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen_parameter_ids:
            continue

        parameter_id = id(parameter)
        groups = (context_group, tp_group) if parameter_id in qk_norm_parameter_ids else (context_group,)
        handle = parameter.register_post_accumulate_grad_hook(_make_grad_sync_hook(groups))
        handles.append((parameter_id, handle))
        seen_parameter_ids.add(parameter_id)

    model._long_context_grad_sync_configured = True
    return handles


def _make_grad_sync_hook(groups: tuple[dist.ProcessGroup, ...]):
    state: dict[str, Any] = {}

    @torch.no_grad()
    def hook(parameter: torch.Tensor) -> None:
        grad = parameter.grad
        if grad is None:
            state.clear()
            return

        local_grad = _local_tensor(grad)
        grad_ref = state.get("grad_ref")
        if grad_ref is None or grad_ref() is not grad:
            state["grad_ref"] = weakref.ref(grad)
            state["synced"] = torch.zeros_like(local_grad)

        synced = state["synced"]
        delta = local_grad.detach().clone()
        delta.sub_(synced)
        for group in groups:
            dist.all_reduce(delta, op=dist.ReduceOp.SUM, group=group)
        local_grad.copy_(synced + delta)
        synced.copy_(local_grad)

    return hook


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        local_tensor = tensor.to_local()
        wait = getattr(local_tensor, "wait", None)
        if callable(wait):
            local_tensor = wait()
        return local_tensor
    return tensor


def bind_qwen3_vl_parallel_hooks(model: torch.nn.Module) -> torch.nn.Module:
    """Bind recipe-local TP and long-context hooks to the runtime Qwen3-VL class."""
    model_class = model.__class__
    model_class.TP_MODULE_CONFIG = QWEN3_VL_TP_MODULE_CONFIG
    model_class.TP_MODULE_POSTPROCESSORS = QWEN3_VL_TP_MODULE_POSTPROCESSORS
    model_class.SEQUENCE_PARALLEL_MODULE_CONFIG = QWEN3_VL_SEQUENCE_PARALLEL_MODULE_CONFIG
    model_class.SEQUENCE_PARALLEL_SEQUENCE_DIM = 1
    model_class.APPLY_LONG_CONTEXT_ATTENTION = staticmethod(apply_long_context_attention_for_qwen3_vl)
    return model


def build_qwen3_vl_model(model_config: Any):
    """Load the Qwen3-VL model checkpoint and apply the freeze policy.

    Args:
        model_config: Recipe model config with load and freeze settings.

    Returns:
        The initialized Qwen3-VL model.
    """
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    attn_implementation = getattr(model_config, "attn_implementation", None)
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = str(attn_implementation)

    model = AutoModelForImageTextToText.from_pretrained(
        model_config.pretrained_model_name_or_path,
        **load_kwargs,
    )
    apply_freeze_policy(model)
    return bind_qwen3_vl_parallel_hooks(model)
