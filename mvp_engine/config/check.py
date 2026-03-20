from omegaconf import DictConfig, ListConfig, OmegaConf


def check_config(config: DictConfig) -> None:
    """Validate required engine config keys and value types."""
    if not isinstance(config, DictConfig):
        raise TypeError(f"`config` must be DictConfig, got {type(config).__name__}.")

    required_keys = [
        "dev_mode",
        "project.name",
        "project.dir",
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

    backend_kwargs = OmegaConf.select(config, "parallel.backend_kwargs", default={}) or {}
    if not isinstance(backend_kwargs, (dict, DictConfig)):
        raise TypeError("`parallel.backend_kwargs` must be a mapping.")

    mesh_cfg = OmegaConf.select(config, "parallel.mesh", default={}) or {}
    if not isinstance(mesh_cfg, (dict, DictConfig)):
        raise TypeError("`parallel.mesh` must be a mapping.")

    for mesh_type in mesh_cfg.keys():
        if mesh_type not in {"replicate", "shard", "tensor"}:
            raise ValueError(f"Invalid mesh type: {mesh_type}. Supported types are ['replicate', 'shard', 'tensor'].")
        value = mesh_cfg[mesh_type]
        if not isinstance(value, int) or isinstance(value, bool) or value < -1 or value == 0:
            raise ValueError(f"`parallel.mesh.{mesh_type}` must be an integer >= 1 or -1.")

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
