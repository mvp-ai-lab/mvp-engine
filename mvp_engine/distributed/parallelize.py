from typing import Any, Dict

import torch.nn as nn
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
    backend: str = "ddp",
    backend_kwargs: Dict = None,
):
    """Parallelize a neural network model using specified distributed backend.

    Supports multiple parallelization strategies including DistributedDataParallel (DDP)
    and Fully Sharded Data Parallel v2 (FSDP2). The function automatically configures
    backend-specific parameters with sensible defaults.

    Args:
        model: The neural network model to parallelize.
        device_mesh: DeviceMesh object containing device topology. Should have a "dp"
                    dimension for data parallelism.
        backend: Parallelization backend to use. Options:
                - "ddp": DistributedDataParallel (default)
                - "fsdp2": Fully Sharded Data Parallel v2
        backend_kwargs: Backend-specific configuration dictionary.

            For DDP:
                - device_mesh: DeviceMesh for data parallel dimension (auto-set from device_mesh["dp"])
                - Additional kwargs passed to DistributedDataParallel constructor

            For FSDP2:
                - target_classes: List of module class names to wrap with FSDP2 (default: [])
                - mesh: DeviceMesh for data parallel dimension (auto-set from device_mesh["dp"])
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
        >>> device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
        >>> model = MyModel().cuda()
        >>> parallel_model = parallelize_model(model, device_mesh, backend="ddp")
    """

    if backend_kwargs is None:
        backend_kwargs = {}
    elif not isinstance(backend_kwargs, dict):
        # Convert OmegaConf DictConfig or other dict-like objects to dict
        backend_kwargs = dict(backend_kwargs)
    if backend == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        logger.info(f"Wrapping {model.__class__.__name__} with DistributedDataParallel...")
        backend_kwargs = _set_default_kwargs(backend_kwargs, "device_mesh", device_mesh["ddp"])
        parallelized_model = DistributedDataParallel(model, **backend_kwargs)
    elif backend == "fsdp2":
        from mvp_engine.distributed.fsdp2 import parallelize_model_with_fsdp2

        logger.info(f"Wrapping {model.__class__.__name__} with FSDP2...")
        backend_kwargs = _set_default_kwargs(backend_kwargs, "target_classes", [])
        backend_kwargs = _set_default_kwargs(backend_kwargs, "mesh", device_mesh["fsdp2"])
        backend_kwargs = _set_default_kwargs(backend_kwargs, "reshard_after_forward", None)
        backend_kwargs = _set_default_kwargs(backend_kwargs, "mp_policy", None)

        # TODO: support cpu offloading
        # TODO: support custom prefetching strategy

        parallelized_model = parallelize_model_with_fsdp2(model, backend_kwargs)

    else:
        raise NotImplementedError(f"Unsupported parallel backend: {backend}")

    return parallelized_model
