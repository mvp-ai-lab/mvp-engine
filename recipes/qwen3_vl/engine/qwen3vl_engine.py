import torch
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from transformers.models import AutoModel

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger


@ENGINE_REGISTRY.register()
class Qwen3VLEngine(Engine):
    def __init__(self, config):
        super().__init__(config)

    def prepare_dataloader(self, workflow="train"):
        raise NotImplementedError("Should implement this method.")

    def prepare_model(self):
        # 0. Main model
        model = AutoModel.from_pretrained(self.config.model.pretrained or self.config.model.name).to(self.device)
        logger.info(f" - Model name: {model.__class__.__name__}")

        # 1. Freeze modules
        # TODO: implement this.

        # 2. Parallelize model
        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend=self.config.parallel.type,
            backend_kwargs=self.config.parallel.get("backend_kwargs", {}),
        )

        return parallelized_model

    def prepare_optimizer(self):
        return torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.config.optim.lr},
                {"params": self.depth_head.parameters(), "lr": self.config.optim.lr},
                {"params": self.rgb_head.parameters(), "lr": self.config.optim.lr},
            ],
            weight_decay=self.config.optim.weight_decay,
        )

    def prepare_scheduler(self):
        warmup_steps = int(self.config.loop.total_steps * self.config.optim.warmup_ratio)
        scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps)
        scheduler_main = PolynomialLR(self.optimizer, total_iters=self.config.loop.total_steps - warmup_steps, power=2)
        return SequentialLR(self.optimizer, [scheduler_warmup, scheduler_main], milestones=[warmup_steps])
