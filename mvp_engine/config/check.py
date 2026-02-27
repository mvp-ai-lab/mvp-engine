from omegaconf import DictConfig, ListConfig, OmegaConf


def check_config(config: DictConfig) -> None:
    """Validate required engine config keys and value types."""
    if not isinstance(config, DictConfig):
        raise TypeError(f"`config` must be DictConfig, got {type(config).__name__}.")

    required_keys = [
        "dev_mode",
        "project.name",
        "project.dir",
        "parallel.type",
        "optim.gradient_accumulation_steps",
        "optim.mixed_precision",
        "loop.policy",
    ]
    missing = [key for key in required_keys if OmegaConf.select(config, key, default=None) is None]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    if not isinstance(config.dev_mode, bool):
        raise TypeError("`dev_mode` must be a bool.")
    if not isinstance(config.project.name, str):
        raise TypeError("`project.name` must be a str.")
    if not isinstance(config.project.dir, str):
        raise TypeError("`project.dir` must be a str.")

    parallel_type = config.parallel.type
    if parallel_type not in {"ddp", "fsdp2"}:
        raise ValueError(f"`parallel.type` must be one of ['ddp', 'fsdp2'], got: {parallel_type}.")
    if parallel_type == "fsdp2":
        backend_kwargs = OmegaConf.select(config, "parallel.backend_kwargs", default={}) or {}
        if not isinstance(backend_kwargs, (dict, DictConfig)):
            raise TypeError("`parallel.backend_kwargs` must be a mapping.")

        mesh_cfg = OmegaConf.select(config, "parallel.mesh", default={}) or {}
        if not isinstance(mesh_cfg, (dict, DictConfig)):
            raise TypeError("`parallel.mesh` must be a mapping.")
        for mesh_key in ("dp_size", "fsdp2_size", "tp_size"):
            mesh_val = mesh_cfg.get(mesh_key, None)
            if mesh_val is None:
                continue
            if not isinstance(mesh_val, int) or isinstance(mesh_val, bool) or mesh_val < 1:
                raise ValueError(f"`parallel.mesh.{mesh_key}` must be an integer >= 1.")

    mixed_precision = config.optim.mixed_precision
    if mixed_precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"`optim.mixed_precision` must be one of ['fp32', 'fp16', 'bf16'], got: {mixed_precision}.")

    grad_steps = config.optim.gradient_accumulation_steps
    if not isinstance(grad_steps, int) or isinstance(grad_steps, bool) or grad_steps < 1:
        raise ValueError("`optim.gradient_accumulation_steps` must be an integer >= 1.")

    loop_policy = config.loop.policy
    if loop_policy not in {"iter", "epoch"}:
        raise ValueError(f"`loop.policy` must be one of ['iter', 'epoch'], got: {loop_policy}.")
    if loop_policy == "iter":
        total_steps = OmegaConf.select(config, "loop.total_steps", default=None)
        if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps < 1:
            raise ValueError("`loop.total_steps` must be an integer >= 1 when `loop.policy` is 'iter'.")

    log_interval = OmegaConf.select(config, "project.log.interval", default=None)
    if log_interval is not None and (
        not isinstance(log_interval, int) or isinstance(log_interval, bool) or log_interval < 1
    ):
        raise ValueError("`project.log.interval` must be an integer >= 1.")

    log_backends = OmegaConf.select(config, "project.log.backends", default=None)
    if log_backends is not None:
        if not isinstance(log_backends, (list, tuple, ListConfig)):
            raise TypeError("`project.log.backends` must be a list of backend names.")
        invalid_backends = [backend for backend in log_backends if backend not in {"terminal", "file"}]
        if invalid_backends:
            raise ValueError(
                f"`project.log.backends` contains invalid values: {invalid_backends}. "
                "Supported backends are ['terminal', 'file']."
            )
