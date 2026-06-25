"""High-level entrypoints for building MLLM dataset and dataloader pipelines."""

from __future__ import annotations

from functools import partial
from operator import methodcaller
from typing import Any

import torch
from mvp_dataset import Dataset, TorchLoader
from mvp_dataset.core import RuntimeContext

from mvp_engine.distributed import ParallelMesh

from .collator import MLLMBatchCollator
from .guard import MLLMModelInputGuard, MLLMRawRowGuard, MLLMSampleGuard
from .packing import MLLMPackingAssembler
from .sample import MLLMSample
from .spec import MLLMDataSpec, MLLMDistributionSpec
from .types import ModelInputs


class MLLMDataKit:
    """Build runtime MLLM data objects from explicit data specs.

    Recipes use this kit as the stable orchestration boundary: they declare source,
    sample, packing, loader, and distribution specs, then let the kit build the
    mvp-dataset pipeline and TorchLoader wrapper.
    """

    def build_distribution_spec(
        self,
        *,
        parallel_mesh: ParallelMesh,
    ) -> MLLMDistributionSpec:
        """Build distributed placement options for MLLM data pipelines.

        Args:
            parallel_mesh: Parallel mesh used to derive mvp-dataset sharding placement.

        Returns:
            A distribution spec ready to attach to ``MLLMDataSpec``.
        """
        return MLLMDistributionSpec(
            device_mesh=parallel_mesh.device_mesh,
            dp_dims=parallel_mesh.dp.dim_names or None,
        )

    def build_processor(
        self,
        pretrained_model_name_or_path: str,
        *,
        trust_remote_code: bool = True,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        tokenizer_padding_side: str = "right",
        pad_token_fallback_to_eos: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Load a Hugging Face processor and normalize common tokenizer/image settings.

        Args:
            pretrained_model_name_or_path: Hugging Face model id or local processor path.
            trust_remote_code: Passed to ``AutoProcessor.from_pretrained``.
            image_min_pixels: Optional minimum image pixel budget applied to image processors that expose it.
            image_max_pixels: Optional maximum image pixel budget applied to image processors that expose it.
            tokenizer_padding_side: Padding side assigned to ``processor.tokenizer`` when present.
            pad_token_fallback_to_eos: Whether to use EOS as PAD when the tokenizer has no PAD token.
            **kwargs: Additional keyword arguments passed to ``AutoProcessor.from_pretrained``.

        Returns:
            The loaded processor with normalized tokenizer and image-processor settings.
        """
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None and (image_min_pixels is not None or image_max_pixels is not None):
            size = getattr(image_processor, "size", None)
            if hasattr(size, "__setitem__"):
                if image_min_pixels is not None:
                    size["shortest_edge"] = int(image_min_pixels)
                if image_max_pixels is not None:
                    size["longest_edge"] = int(image_max_pixels)
            if image_min_pixels is not None and hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = int(image_min_pixels)
            if image_max_pixels is not None and hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = int(image_max_pixels)

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = tokenizer_padding_side
            if pad_token_fallback_to_eos and tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

        return processor

    def build_dataset(self, spec: MLLMDataSpec) -> Dataset:
        """Build the source, guard, sample, packing, ref-resolution, and model-input pipeline.

        Args:
            spec: Complete data declaration for one MLLM dataset pipeline.

        Returns:
            An mvp-dataset ``Dataset`` that yields finalized packed model-input dictionaries.
        """
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
            shuffle_mode="chunk",
        )
        dataset = dataset.assemble(MLLMRawRowGuard)
        dataset = dataset.map(partial(MLLMSample.from_row, sample_spec=spec.sample))
        dataset = dataset.assemble(MLLMSampleGuard)
        assembler_cls = spec.packing.assembler_cls or MLLMPackingAssembler
        dataset = dataset.assemble(partial(assembler_cls, spec.packing))
        if spec.source.resolve_refs and spec.source.ref_columns:
            dataset = dataset.resolve_ref(ref_names=spec.source.ref_columns)
        dataset = dataset.map(methodcaller("to_model_inputs", load_media=source.resolve_refs))
        dataset = dataset.assemble(
            partial(
                MLLMModelInputGuard,
                ignore_index=spec.sample.tokenization_handler.ignore_index,
                verbose=False,
            )
        )
        return dataset

    def build_collator(self, spec: MLLMDataSpec) -> MLLMBatchCollator:
        """Build the standard token-padding and media-collation function.

        Args:
            spec: Complete data declaration whose sample handlers provide tokenizer and media behavior.

        Returns:
            A collator suitable for packed MLLM model-input dictionaries.

        Raises:
            ValueError: If the processor tokenizer does not expose ``pad_token_id``.
        """
        processor = spec.sample.tokenization_handler.processor
        tokenizer = getattr(processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("Processor tokenizer must expose a pad_token_id to build an MLLM collator.")

        return MLLMBatchCollator(
            pad_token_id=pad_token_id,
            media_handler=spec.sample.media_handler,
            ignore_index=spec.sample.tokenization_handler.ignore_index,
        )

    def build_dataloader(
        self,
        dataset: Dataset,
        spec: MLLMDataSpec,
        *,
        device: torch.device | None = None,
    ):
        """Wrap an mvp-dataset dataset in a TorchLoader batch pipeline.

        Args:
            dataset: Dataset produced by ``build_dataset`` or an equivalent packed-input dataset.
            spec: Complete data declaration whose loader spec controls batching.
            device: Optional target device used only to choose the default ``pin_memory`` value.

        Returns:
            A batched TorchLoader iterator.
        """
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
        """Move tensor values in one model-input batch to the target device.

        Args:
            batch: Collated model-input dictionary.
            device: Destination torch device.

        Returns:
            A shallow-copied batch with tensor values moved to ``device``.
        """
        batch_on_device = {}
        for key, value in batch.items():
            batch_on_device[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return batch_on_device
