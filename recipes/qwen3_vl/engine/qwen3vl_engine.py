import torch
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from transformers.models import AutoModel

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import is_main_process
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger
from mvp_engine.utils.misc import calculate_model_size

from ..dataset.jsonl import build_jsonl_dataloader
from ..utils.model import freeze_module


@ENGINE_REGISTRY.register()
class Qwen3VLEngine(Engine):
    def __init__(self, config):
        super().__init__(config)

    def prepare_dataloader(self, workflow="train"):
        if workflow == "train":
            if self.config.data.dataset_type == "jsonl":
                dataloader = build_jsonl_dataloader(
                    jsonl_path=self.config.data.data_path,
                    batch_size=self.config.data.batch_size,
                    num_workers=self.config.data.num_workers,
                    shuffle_buffer=self.config.data.shuffle_buffer,
                )
            else:
                raise NotImplementedError(f"Unsupported dataset type: {self.config.data.dataset_type}")
            return dataloader

    def prepare_model(self):
        # 0. Main model
        model = AutoModel.from_pretrained(self.config.model.pretrained or self.config.model.name).to(self.device)
        logger.info(f" - Model name: {model.__class__.__name__}")

        # 1. Freeze modules
        projector_keywords = (
            "projector",
            "mm_projector",
            "multi_modal_projector",
            "merger",
        )

        visual_module = getattr(model, "visual", None) or getattr(model, "vision_tower", None)
        if visual_module is None:
            model_container = getattr(model, "model", None)
            visual_module = getattr(model_container, "visual", None) or getattr(model_container, "vision_tower", None)

        language_module = getattr(model, "language_model", None)
        if language_module is None:
            model_container = getattr(model, "model", None)
            language_module = getattr(model_container, "language_model", None)

        lm_head_module = getattr(model, "lm_head", None)

        if self.config.model.freeze_vit:
            frozen = 0
            if visual_module is not None:
                frozen += freeze_module(visual_module, exclude_keywords=projector_keywords)
            logger.info(f" - Freeze ViT params: {frozen / 1e9:.4f} B")

        if self.config.model.freeze_projector:
            frozen = 0
            if visual_module is not None:
                frozen += freeze_module(visual_module, include_keywords=projector_keywords)
            for attr in (
                "projector",
                "mm_projector",
                "multi_modal_projector",
                "merger",
            ):
                module = getattr(model, attr, None)
                if module is not None:
                    frozen += freeze_module(module)
            logger.info(f" - Freeze projector params: {frozen / 1e9:.4f} B")

        if self.config.model.freeze_llm:
            frozen = 0
            if language_module is not None:
                frozen += freeze_module(language_module)
            if lm_head_module is not None:
                frozen += freeze_module(lm_head_module)
            logger.info(f" - Freeze LLM params: {frozen / 1e9:.4f} B")

        # 2. Parallelize model
        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend=self.config.parallel.type,
            backend_kwargs=self.config.parallel.get("backend_kwargs", {}),
        )

        # 3. Calculate model size in B
        if is_main_process():
            model_size, trainable_size = calculate_model_size(parallelized_model)
            logger.info(f" - Model size: {model_size / 1e9:.4f} B")
            logger.info(f" - Trainable model size: {trainable_size / 1e9:.4f} B")

        return parallelized_model

    def prepare_optimizer(self):
        return torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.config.optim.lr},
            ],
            weight_decay=self.config.optim.weight_decay,
        )

    def prepare_scheduler(self):
        warmup_steps = int(self.config.loop.total_steps * self.config.optim.warmup_ratio)
        scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps)
        scheduler_main = PolynomialLR(
            self.optimizer,
            total_iters=self.config.loop.total_steps - warmup_steps,
            power=2,
        )
        return SequentialLR(
            self.optimizer,
            [scheduler_warmup, scheduler_main],
            milestones=[warmup_steps],
        )

    def train_pre_step(self, data: dict):
        return data
