from typing import Any, Dict

import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy

from mvp_engine.distributed.utils import (
    MESH_DIM_SHARD,
    MESH_DIM_TENSOR,
    get_context_parallel_size,
    get_mesh_dim_size,
    get_mesh_shape,
    get_replicate_mesh,
    get_sharded_data_parallel_mesh,
    get_tensor_parallel_mesh,
)
from mvp_engine.utils.log import logger


def _set_default_kwargs(kwargs: Dict, key: str, value: Any) -> Dict:
    """Set default value for a key in kwargs dictionary if not already present.

    Args:
        kwargs: The dictionary to update.
        key: The key to check and potentially set.
        value: The default value to set if key is not present.

    Returns:
        The updated kwargs dictionary.
    """
    if key not in kwargs:
        kwargs[key] = value
    return kwargs


def parallelize_model(
    model: nn.Module,
    device_mesh: DeviceMesh,
    backend_kwargs: Dict = None,
) -> nn.Module:
    """Parallelize a neural network model using specified distributed backend.

    Supports multiple parallelization strategies including DistributedDataParallel (DDP)
    and Fully Sharded Data Parallel v2 (FSDP2). The function automatically configures
    backend-specific parameters with sensible defaults.

    Args:
        model: The neural network model to parallelize.
        device_mesh: DeviceMesh object containing device topology.
        backend_kwargs: Backend-specific configuration dictionary.

            For DDP:
                - device_mesh: DeviceMesh for data parallel dimension (auto-set from the first dim of the device_mesh)
                - Additional kwargs passed to DistributedDataParallel constructor

            For FSDP2:
                - target_classes: List of module class names to wrap with FSDP2 (default: [])
                - mesh: DeviceMesh for data parallel dimension (auto-set from first two dims of the device_mesh)
                - reshard_after_forward: Resharding strategy (default: None)
                - mp_policy: Mixed precision policy (default: None)
                - high_precision_modules / high_precision_mp_policy:
                  parsed and handled inside fsdp2.py
                - Additional kwargs passed to fully_shard()

            For Tensor Parallel and Sequence Parallel (if tensor mesh is active):
                - sequence_parallel: Enables sequence parallel layouts on the tensor mesh before FSDP2 (default: False)

            For Long-Context Attention (if context mesh is active):
                - long_context: Initializes yunchang USP/Ring attention before TP/FSDP2 and installs context grad sync.

    Returns:
        The parallelized model wrapped with the specified backend.

    Raises:
        NotImplementedError: If the specified backend is not supported.

    Example:
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("replicate",))
        >>> model = MyModel().cuda()
        >>> parallel_model = parallelize_model(model, device_mesh)
    """

    if backend_kwargs is None:
        backend_kwargs = {}

    # Hydra configs are often DictConfig/ListConfig objects in struct mode.
    # Convert them to plain Python containers before we inject backend defaults.
    if OmegaConf.is_config(backend_kwargs):
        backend_kwargs = OmegaConf.to_container(backend_kwargs, resolve=True)

    backend_kwargs = dict(backend_kwargs)
    tensor_size = get_mesh_dim_size(device_mesh, MESH_DIM_TENSOR)
    shard_size = get_mesh_dim_size(device_mesh, MESH_DIM_SHARD)
    context_size = get_context_parallel_size(device_mesh)

    sequence_parallel = bool(backend_kwargs.pop("sequence_parallel", False))
    long_context_kwargs = backend_kwargs.pop("long_context", {}) or {}
    from mvp_engine.distributed.cp import (
        install_context_grad_sync,
        is_long_context_enabled,
        prepare_long_context_attention,
    )

    long_context = is_long_context_enabled(long_context_kwargs)
    if sequence_parallel and tensor_size <= 1:
        raise ValueError("Sequence parallel requires an active tensor mesh with parallel.mesh.tensor > 1.")
    if long_context and sequence_parallel:
        raise ValueError("Long-context attention and tensor-mesh sequence_parallel cannot be enabled together.")
    if long_context and context_size <= 1:
        raise ValueError("Long-context attention requires parallel.mesh.context > 1.")
    if context_size > 1 and not long_context:
        raise ValueError("parallel.mesh.context > 1 requires parallel.backend_kwargs.long_context.enabled=true.")

    if shard_size * tensor_size * context_size == 1:
        # For Pure DDP: [N, 1, 1]
        from torch.nn.parallel import DistributedDataParallel

        logger.info(f"Wrapping {model.__class__.__name__} with DistributedDataParallel...")
        backend_kwargs = backend_kwargs.get("ddp", {})
        backend_kwargs = _set_default_kwargs(backend_kwargs, "device_mesh", get_replicate_mesh(device_mesh))
        parallelized_model = DistributedDataParallel(model, **backend_kwargs)
    else:
        # For FSDP2: [N, M, ...]
        if shard_size == 1 and (tensor_size > 1 or context_size > 1):
            raise ValueError(
                f"Invalid device mesh shape {get_mesh_shape(device_mesh)}. "
                "Tensor/sequence/long-context parallel should be used with FSDP rather than the pure DDP"
            )

        from mvp_engine.distributed.fsdp2 import parallelize_model_with_fsdp2

        fsdp2_mesh = get_sharded_data_parallel_mesh(device_mesh)
        tp_mesh = None
        if tensor_size > 1:
            tp_mesh = get_tensor_parallel_mesh(device_mesh)

        if long_context:
            logger.info(f"Preparing {model.__class__.__name__} for Long-Context Attention...")
            prepare_long_context_attention(model, device_mesh, long_context_kwargs)

        if tp_mesh is not None and tp_mesh.size() > 1:
            from mvp_engine.distributed.tp import parallelize_model_with_tensor_parallel

            logger.info(f"Wrapping {model.__class__.__name__} with Tensor Parallel...")

            parallelize_model_with_tensor_parallel(
                model,
                tp_mesh,
                sequence_parallel=sequence_parallel,
            )

        parallelized_model = model
        if fsdp2_mesh is not None and fsdp2_mesh.size() > 1:
            logger.info(f"Wrapping {model.__class__.__name__} with FSDP2...")
            backend_kwargs = backend_kwargs.get("fsdp2", {})
            backend_kwargs = _set_default_kwargs(backend_kwargs, "target_classes", [])
            backend_kwargs = _set_default_kwargs(backend_kwargs, "mesh", fsdp2_mesh)
            backend_kwargs = _set_default_kwargs(backend_kwargs, "reshard_after_forward", True)
            backend_kwargs = _set_default_kwargs(
                backend_kwargs,
                "mp_policy",
                {
                    "param_dtype": "bfloat16",
                    "reduce_dtype": "float32",
                    "buffer_dtype": "bfloat16",
                },
            )

            # Map the config-level boolean to PyTorch's composable FSDP2 policy object.
            if backend_kwargs.pop("offload_policy", False):
                backend_kwargs["offload_policy"] = CPUOffloadPolicy()

            parallelized_model = parallelize_model_with_fsdp2(model, backend_kwargs)

        if long_context and bool(long_context_kwargs.get("grad_sync", True)):
            install_context_grad_sync(parallelized_model, device_mesh)

    return parallelized_model
