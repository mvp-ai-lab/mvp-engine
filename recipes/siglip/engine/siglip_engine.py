from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoTokenizer, SiglipConfig, SiglipTextConfig
from transformers.utils.logging import disable_progress_bar
from webdataset import WebLoader

from mvp_engine.dataset.webdataset import WebDatasetBuilder
from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger

from ..dataset import caption_dataset as cpt_dset
from ..evals.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
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
        if workflow == "evaluate":
            imagenet_path = OmegaConf.select(self.config, "eval.imagenet_path", default=None)
            if not imagenet_path:
                logger.warning("Skip dataloader preparation for workflow: evaluate (missing eval.imagenet_path)")
                return None

            eval_transform = transforms.Compose(
                [
                    transforms.Resize(int(self.config.data.resize), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(self.config.data.resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ]
            )
            dataset = ImageFolder(imagenet_path, transform=eval_transform)
            sampler = None
            if dist.is_available() and dist.is_initialized():
                sampler = DistributedSampler(dataset, shuffle=True)

            return DataLoader(
                dataset,
                batch_size=OmegaConf.select(self.config, "eval.batch_size", default=self.config.data.batch_size),
                num_workers=OmegaConf.select(self.config, "eval.num_workers", default=self.config.data.num_workers),
                sampler=sampler,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            )
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

        # 7. Load from a checkpoint if specified
        self.model = parallelized_model
        load_from_cfg = OmegaConf.select(self.config, "model.load_from", default=None)
        if load_from_cfg and load_from_cfg.path:
            self._load_model_weights(load_from_cfg.path)

        return parallelized_model

    def prepare_optimizer(self):
        optimizer = create_optimizer_v2(
            self.model, self.config.optim.name, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay
        )

        return optimizer

    def prepare_scheduler(self):
        warmup_steps = int(self.config.loop.total_steps * self.config.optim.warmup_ratio)
        scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps)
        scheduler_main = PolynomialLR(self.optimizer, total_iters=self.config.loop.total_steps - warmup_steps, power=2)
        return SequentialLR(self.optimizer, [scheduler_warmup, scheduler_main], milestones=[warmup_steps])

    def run_train(self):
        self.model.train()

        return super().run_train()

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
            loss = self.model(
                input_ids=data["text_input"],
                pixel_values=data["pixel_input"],
                attention_mask=data.get("attention_mask"),
                return_loss=True,
            )
        return {
            "loss": loss["loss"],
            "logs": {
                "train/loss": loss["loss"].item(),
            },
        }

    def train(self) -> None:
        if OmegaConf.select(self.config, "test_only", default=False):
            self.evaluate()
            return
        super().train()

    def train_pre_step(self, data: List) -> Dict:
        """Preprocess the input data before training step."""
        batch = {}
        pixel_values, text_inputs = data
        for k, v in text_inputs.items():
            text_inputs[k] = v.to(self.device, non_blocking=True)

        pixel_values = pixel_values.to(self.device, non_blocking=True)
        batch["text_input"] = text_inputs["input_ids"]
        batch["attention_mask"] = text_inputs.get("attention_mask")
        batch["pixel_input"] = pixel_values

        return batch

    def _load_model_weights(self, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        model_path = ckpt_path / "model.pt" if ckpt_path.is_dir() else ckpt_path
        state_dict = torch.load(model_path, map_location="cpu")
        target_model = self.model.module if hasattr(self.model, "module") else self.model
        target_model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {model_path}")

    def _build_zero_shot_classifier(self, class_to_idx: Dict[str, int]) -> torch.Tensor:
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
        text_features = []
        model = self.unwrapped_model
        aligned_class_names = []
        for k, v in class_to_idx.items():
            aligned_class_names.append(imagenet_classnames[int(k)])

        with torch.no_grad():
            for class_name in aligned_class_names:
                prompts = [template(class_name) for template in openai_imagenet_template]
                tokenized = tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                tokenized = {k: v.to(self.device, non_blocking=True) for k, v in tokenized.items()}
                with torch.autocast(
                    device_type=self.device_type,
                    dtype=self.dtype,
                    enabled=self.dtype != torch.float32,
                ):
                    class_embeddings = model.get_text_features(
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized.get("attention_mask"),
                    )
                class_embeddings = class_embeddings / class_embeddings.norm(p=2, dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                text_features.append(class_embeddings)

        return torch.stack(text_features, dim=0)

    @staticmethod
    def _accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> List[torch.Tensor]:
        max_k = max(topk)
        pred = logits.topk(max_k, dim=1, largest=True, sorted=True).indices.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum() for k in topk]

    def before_evaluate(self) -> None:
        logger.log_config(self.config)
        logger.info("Building evaluation DataLoader...")
        self.evaluate_loader = self.prepare_dataloader("evaluate")
        if self.evaluate_loader is None:
            raise ValueError("Evaluation requested but eval.imagenet_path is not configured.")

        logger.info("Building Model...")
        self.model = self.prepare_model()

        checkpoint_path = OmegaConf.select(self.config, "eval.checkpoint_path", default=None)
        if checkpoint_path:
            self._load_model_weights(checkpoint_path)

        self.model.eval()

    @torch.no_grad()
    def run_evaluate(self) -> Dict[str, float]:
        classifier = self._build_zero_shot_classifier(self.evaluate_loader.dataset.class_to_idx)
        model = self.unwrapped_model

        local_top1 = torch.tensor(0.0, device=self.device)
        local_top5 = torch.tensor(0.0, device=self.device)
        local_count = torch.tensor(0.0, device=self.device)

        for images, target in self.evaluate_loader:
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device_type,
                dtype=self.dtype,
                enabled=self.dtype != torch.float32,
            ):
                image_features = model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                logits = torch.matmul(classifier, image_features.t())
                logits = logits * model.logit_scale.exp() + model.logit_bias

            acc1, acc5 = self._accuracy(logits.T, target, topk=(1, 5))
            local_top1 += acc1
            local_top5 += acc5
            local_count += target.shape[0]

        stats = torch.stack([local_top1, local_top5, local_count])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        total_count = max(stats[2].item(), 1.0)
        return {
            "eval/imagenet_top1": stats[0].item() / total_count,
            "eval/imagenet_top5": stats[1].item() / total_count,
            "eval/num_samples": stats[2].item(),
        }

    def after_evaluate(self, metrics: Dict[str, float]) -> None:
        logger.log_metrics(metrics, step=0)
        logger.info(
            "Evaluation finished: "
            f"top1={metrics['eval/imagenet_top1']:.4f}, "
            f"top5={metrics['eval/imagenet_top5']:.4f}, "
            f"samples={int(metrics['eval/num_samples'])}"
        )
        logger.destroy()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.before_evaluate()
        metrics = self.run_evaluate()
        self.after_evaluate(metrics)
        return metrics
