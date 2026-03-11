from typing import Any, Dict

import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed.device_mesh import DeviceMesh

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
):
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

    if device_mesh["shard"].shape[0] * device_mesh["tensor"].shape[0] == 1:
        # For Pure DDP: [N, 1, 1]
        from torch.nn.parallel import DistributedDataParallel

        logger.info(f"Wrapping {model.__class__.__name__} with DistributedDataParallel...")
        backend_kwargs = backend_kwargs.get("ddp", {})
        backend_kwargs = _set_default_kwargs(backend_kwargs, "device_mesh", device_mesh["replicate"])
        parallelized_model = DistributedDataParallel(model, **backend_kwargs)
    else:
        # For FSDP2: [N, M, ...]
        if device_mesh["shard"].size() == 1 and device_mesh["tensor"].size() > 1:
            raise ValueError(
                f"Invalid device mesh shape {device_mesh.shape}. Tensor parallel should be used with FSDP rather than the pure DDP"
            )

        from mvp_engine.distributed.fsdp2 import parallelize_model_with_fsdp2

        fsdp2_mesh = device_mesh["replicate", "shard"]
        tp_mesh = device_mesh["tensor"] if device_mesh.ndim > 2 else None

        if tp_mesh is not None and tp_mesh.size() > 1:
            from mvp_engine.distributed.tp import parallelize_model_with_tensor_parallel

            logger.info(f"Wrapping {model.__class__.__name__} with Tensor Parallel...")

            applied = parallelize_model_with_tensor_parallel(
                model,
                tp_mesh,
            )
            logger.info(f"Applied Tensor Parallel to {len(applied)} modules on {model.__class__.__name__}.")

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

            # TODO: support cpu offloading
            # TODO: support custom prefetching strategy

            parallelized_model = parallelize_model_with_fsdp2(model, backend_kwargs)

    return parallelized_model
