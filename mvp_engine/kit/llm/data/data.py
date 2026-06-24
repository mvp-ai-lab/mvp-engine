"""High-level entrypoints for building text-only LM dataset and dataloader pipelines."""

from __future__ import annotations

from functools import partial
from operator import methodcaller
from typing import Any

import torch
from mvp_dataset import Dataset, TorchLoader
from mvp_dataset.core import RuntimeContext

from mvp_engine.distributed import ParallelMesh

from .collator import LLMBatchCollator
from .guard import LLMModelInputGuard, LLMRawRowGuard, LLMSampleGuard
from .packing import LLMPackingAssembler
from .sample import LLMSampleAssembler
from .spec import LLMDataSpec, LLMDistributionSpec
from .types import ModelInputs

IGNORE_INDEX = -100


class LLMDataKit:
    """Build runtime text-only LM data objects from explicit data specs."""

    def build_distribution_spec(
        self,
        *,
        parallel_mesh: ParallelMesh,
    ) -> LLMDistributionSpec:
        """Build distributed placement options for LLM data pipelines."""
        return LLMDistributionSpec(
            device_mesh=parallel_mesh.device_mesh,
            dp_dims=parallel_mesh.dp.dim_names or None,
        )

    def build_tokenizer(
        self,
        pretrained_model_name_or_path: str,
        *,
        trust_remote_code: bool = True,
        padding_side: str = "right",
        pad_token_fallback_to_eos: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Load a Hugging Face tokenizer and normalize pad/padding settings."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        tokenizer.padding_side = padding_side
        if pad_token_fallback_to_eos and tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def build_dataset(self, spec: LLMDataSpec) -> Dataset:
        """Build the source, guard, sample, packing, and model-input pipeline."""
        source = spec.source
        distribution = spec.distribution
        if distribution is None or distribution.device_mesh is None or distribution.dp_dims is None:
            context = RuntimeContext.from_runtime(seed=source.seed)
        else:
            context = RuntimeContext.from_runtime(
                seed=source.seed,
                device_mesh=distribution.device_mesh,
                dp_dims=distribution.dp_dims,
            )

        dataset = Dataset.from_source(
            source.dataset_source,
            source.dataset_path,
            context=context,
            resample=source.resample,
            shuffle_mode=source.shuffle_mode,
        )
        dataset = dataset.assemble(partial(LLMRawRowGuard, schema_handler=spec.sample.schema_handler))
        dataset = dataset.assemble(partial(LLMSampleAssembler, spec.sample))
        dataset = dataset.assemble(LLMSampleGuard)
        assembler_cls = spec.packing.assembler_cls or LLMPackingAssembler
        dataset = dataset.assemble(partial(assembler_cls, spec.packing))
        dataset = dataset.map(methodcaller("to_model_inputs"))
        dataset = dataset.assemble(
            partial(
                LLMModelInputGuard,
                ignore_index=spec.sample.tokenization_handler.ignore_index,
                verbose=False,
            )
        )
        return dataset

    def build_collator(
        self,
        spec: LLMDataSpec,
    ) -> LLMBatchCollator:
        """Build the standard token-padding collation function."""
        tokenizer = spec.sample.tokenization_handler.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must expose a pad_token_id to build an LLM collator.")
        return LLMBatchCollator(
            pad_token_id=int(pad_token_id),
            ignore_index=spec.sample.tokenization_handler.ignore_index,
        )

    def build_dataloader(
        self,
        dataset: Dataset,
        spec: LLMDataSpec,
        *,
        device: torch.device | None = None,
    ):
        """Wrap an mvp-dataset dataset in a TorchLoader batch pipeline."""
        loader = spec.loader
        pin_memory = loader.pin_memory
        if pin_memory is None:
            pin_memory = device is not None and device.type in {"cuda", "npu"}
        torch_loader = TorchLoader(
            dataset,
            num_workers=loader.num_workers,
            pin_memory=pin_memory,
            persistent_workers=loader.persistent_workers,
            multiprocessing_context=loader.multiprocessing_context,
        )
        return torch_loader.batch(
            batch_size=loader.batch_size,
            drop_last=loader.drop_last,
            collate_fn=self.build_collator(spec),
        )

    def to_device(self, batch: ModelInputs, device: torch.device) -> ModelInputs:
        """Move all tensor fields of a batch to the target device."""
        batch_on_device: dict[str, Any] = {}
        for key, value in batch.items():
            batch_on_device[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return batch_on_device


__all__ = [
    "IGNORE_INDEX",
    "LLMBatchCollator",
    "LLMDataKit",
    "ModelInputs",
]
