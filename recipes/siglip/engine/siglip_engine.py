from functools import partial
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from transformers import AutoTokenizer, SiglipConfig, SiglipTextConfig
from transformers.utils.logging import disable_progress_bar
from webdataset import WebLoader

from mvp_engine.dataset.webdataset import WebDatasetBuilder
from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger

from ..dataset import caption_dataset as cpt_dset
from ..model.modeling_siglip import Qwen2_5_VLVisionConfig, SiglipModel


@ENGINE_REGISTRY.register()
class SigLipEngine(Engine):
    def __init__(self, config):
        super().__init__(config)
        disable_progress_bar()

    def prepare_dataloader(self, workflow="train"):
        if workflow == "train":
            train_transform = cpt_dset.get_train_transforms(self.config.data)
            tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")

            dataset = WebDatasetBuilder(self.config.data.data_path).build(
                batch_size=self.config.data.batch_size,
                make_sample_fn=partial(cpt_dset.decode_data, transform=train_transform),
                shuffle_buffer=self.config.data.shuffle_buffer,
                collate_fn=partial(cpt_dset.collate_fn, tokenizer=tokenizer),
            )
            dataloader = WebLoader(dataset, batch_size=None, num_workers=self.config.data.num_workers)

            return dataloader
        else:
            logger.warning(f"Skip dataloader preparation for workflow: {workflow}")

    def prepare_model(self):
        # 0. Main model
        text_config = SiglipTextConfig(
            hidden_size=self.config.model.text_config.hidden_size,
            intermediate_size=self.config.model.text_config.intermediate_size,
        )
        vision_config = Qwen2_5_VLVisionConfig(
            depth=self.config.model.vision_config.depth,
            hidden_size=self.config.model.vision_config.hidden_size,
            num_heads=self.config.model.vision_config.num_heads,
            intermediate_size=self.config.model.vision_config.intermediate_size,
            temporal_patch_size=self.config.model.vision_config.temporal_patch_size,
            hidden_act=self.config.model.vision_config.hidden_act,
            fullatt_block_indexes=self.config.model.vision_config.fullatt_block_indexes,
        )

        siglip_config = SiglipConfig.from_text_vision_configs(text_config, vision_config)

        model = SiglipModel(siglip_config).to(self.device)
        logger.info(f" - Model name: {model.__class__.__name__}")

        # 5. Parallelize student and teacher
        if self.config.parallel.type in ["ddp", "fsdp2"]:
            parallelized_model = parallelize_model(
                model,
                device_mesh=self.device_mesh,
                backend=self.config.parallel.type,
                backend_kwargs=self.config.parallel.get("backend_kwargs", {}),
            )
        else:
            raise NotImplementedError(f"Parallel type {self.config.parallel.type} not implemented.")

        # 5. Calculate model size in B
        model_size = sum(p.numel() for p in parallelized_model.parameters())
        logger.info(f" - Model size: {model_size / 1e9:.4f} B")
        trainable_size = sum(p.numel() for p in parallelized_model.parameters() if p.requires_grad)
        logger.info(f" - Trainable model size: {trainable_size / 1e9:.4f} B")

        # 6. Compile model
        parallelized_model = torch.compile(
            parallelized_model, backend=self.config.optim.compile_backend, mode=self.config.optim.compile_mode
        )

        # 7. Load from a checkpoint if specified
        self.model = parallelized_model
        load_from_cfg = OmegaConf.select(self.config, "model.load_from", default=None)
        if load_from_cfg and load_from_cfg.path:
            self.load(
                ckpt_path=load_from_cfg.path,
                only_model=load_from_cfg.only_model,
                reset_teacher=load_from_cfg.reset_teacher,
            )

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
        scheduler_main = PolynomialLR(self.optimizer, total_iters=self.config.loop.total_steps - warmup_steps, power=2)
        return SequentialLR(self.optimizer, [scheduler_warmup, scheduler_main], milestones=[warmup_steps])

    def run_train(self):
        self.model.train()

        return super().run_train()

    def train_pre_step(self, data: List) -> Dict:
        """Preprocess the input data before training step."""
        batch = {}
        pixel_values, text_inputs = data
        for k, v in text_inputs.items():
            text_inputs[k] = v.to(self.device, non_blocking=True)

        pixel_values = pixel_values.to(self.device, non_blocking=True)
        batch["text_input"] = text_inputs["input_ids"]
        batch["pixel_input"] = pixel_values

        return batch

    def train_one_step(self, data: Dict) -> Dict:
        """Execute the model forward to get outputs.

        Args:
            data: Preprocessed input batch.

        Returns:
            Dict containing 'loss' and 'logs' keys from model forward.
        """

        # Forward pass with mixed precision autocast
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            loss = self.model(data["text_input"], data["pixel_input"], return_loss=True)

        return {
            "loss": loss,
        }
