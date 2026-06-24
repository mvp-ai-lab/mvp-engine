from typing import Any, Dict

import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed.fsdp import CPUOffloadPolicy

from mvp_engine.distributed.parallel_mesh import ParallelMesh
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
    parallel_mesh: ParallelMesh,
    backend_kwargs: Dict = None,
) -> nn.Module:
    """Parallelize a neural network model using specified distributed backend.

    Supports multiple parallelization strategies including DistributedDataParallel (DDP)
    and Fully Sharded Data Parallel v2 (FSDP2). The function automatically configures
    backend-specific parameters with sensible defaults.

    Args:
        model: The neural network model to parallelize.
        parallel_mesh: Parallel mesh object containing device topology and role accessors.
        backend_kwargs: Backend-specific configuration dictionary.

            For DDP:
                - device_mesh: DeviceMesh for data parallel dimension (auto-set from parallel_mesh.ddp.mesh)
                - Additional kwargs passed to DistributedDataParallel constructor

            For FSDP2:
                - target_classes: List of module class names to wrap with FSDP2 (default: [])
                - mesh: DeviceMesh for data parallel dimension (auto-set from parallel_mesh.fsdp.mesh)
                - reshard_after_forward: Resharding strategy (default: None)
                - mp_policy: Mixed precision policy (default: None)
                - high_precision_modules / high_precision_mp_policy:
                  parsed and handled inside fsdp2.py
                - Additional kwargs passed to fully_shard()

            For Tensor Parallel and Sequence Parallel (if tensor mesh is active):
                - sequence_parallel: Enables sequence parallel layouts on the tensor mesh before FSDP2 (default: False)

    Returns:
        The parallelized model wrapped with the specified backend.

    Raises:
        NotImplementedError: If the specified backend is not supported.
    """

    if backend_kwargs is None:
        backend_kwargs = {}

    # Hydra configs are often DictConfig/ListConfig objects in struct mode.
    # Convert them to plain Python containers before we inject backend defaults.
    if OmegaConf.is_config(backend_kwargs):
        backend_kwargs = OmegaConf.to_container(backend_kwargs, resolve=True)

    backend_kwargs = dict(backend_kwargs)
    ddp_role = parallel_mesh.ddp
    fsdp_role = parallel_mesh.fsdp
    tp_role = parallel_mesh.tp
    shard_role = parallel_mesh.role("shard") if parallel_mesh.has_dim("shard") else None

    sequence_parallel = bool(backend_kwargs.pop("sequence_parallel", False))
    if sequence_parallel and not tp_role.active:
        raise ValueError("Sequence parallel requires an active tensor mesh with parallel.mesh.tensor > 1.")

    shard_active = shard_role is not None and shard_role.active
    if not shard_active and not tp_role.active:
        # For Pure DDP: [N, 1, 1]
        from torch.nn.parallel import DistributedDataParallel

        logger.info(f"Wrapping {model.__class__.__name__} with DistributedDataParallel...")
        backend_kwargs = backend_kwargs.get("ddp", {})
        backend_kwargs = _set_default_kwargs(backend_kwargs, "device_mesh", ddp_role.mesh)
        parallelized_model = DistributedDataParallel(model, **backend_kwargs)
    else:
        # For FSDP2: [N, M, ...]
        if not shard_active and tp_role.active:
            raise ValueError(
                f"Invalid device mesh shape {parallel_mesh.device_mesh.shape}. "
                "Tensor/sequence parallel should be used with FSDP rather than the pure DDP"
            )

        from mvp_engine.distributed.fsdp2 import parallelize_model_with_fsdp2

        fsdp2_mesh = fsdp_role.mesh
        tp_mesh = tp_role.mesh

        if tp_mesh is not None and tp_role.active:
            from mvp_engine.distributed.tp import parallelize_model_with_tensor_parallel

            logger.info(f"Wrapping {model.__class__.__name__} with Tensor Parallel...")

            parallelize_model_with_tensor_parallel(
                model,
                tp_mesh,
                sequence_parallel=sequence_parallel,
            )

        parallelized_model = model
        if fsdp2_mesh is not None and shard_active:
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

    return parallelized_model
