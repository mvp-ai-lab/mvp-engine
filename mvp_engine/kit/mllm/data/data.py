"""Reusable MLLM dataset, dataloader, and collation utilities."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from mvp_dataset import Dataset, TorchLoader
from mvp_dataset.core import RuntimeContext
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from .guard import DataGuard
from .packing import PackingAssembler, PackingOptions, finalize_packed_samples
from .process import IMAGE_PLACEHOLDER
from .process import (
    convert_images_to_pixel_values as convert_images_to_pixel_values_impl,
)
from .process import process_sample as process_sample_impl
from .types import ModelInputs


class MLLMDataKit:
    """Reusable data utilities for standard MLLM recipes."""

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
        """Load a Hugging Face processor and normalize tokenizer/image settings."""
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None and (image_min_pixels is not None or image_max_pixels is not None):
            size = getattr(image_processor, "size", None)
            if isinstance(size, dict):
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

    def build_dataset(
        self,
        dataset_path: str,
        *,
        processor: Any,
        max_seq_len: int,
        resample: bool = True,
        resolve_refs: bool = True,
        ref_columns: list[str] | tuple[str, ...] = ("images",),
        seed: int = 42,
        packing: PackingOptions = PackingOptions(),
        thinking_mode: bool | None | str = True,
    ) -> Dataset:
        """Build an always-packed MLLM training dataset pipeline."""
        context = RuntimeContext.from_runtime(seed=seed)

        dataset = Dataset.from_source(
            "lance",
            dataset_path,
            context=context,
            resample=resample,
            shuffle_mode="fragment_aware",
        )

        dataset = dataset.assemble(
            partial(
                self.build_dataguard,
                check_basic_formats=True,
                check_input_ids=False,
                check_image_sizes=True,
            )
        )

        dataset = dataset.map(
            partial(
                self.process_sample,
                processor=processor,
                max_length=int(max_seq_len),
                thinking_mode=thinking_mode,
            )
        )

        dataset = dataset.assemble(
            partial(
                self.build_dataguard,
                check_basic_formats=False,
                check_input_ids=True,
                check_image_sizes=False,
            )
        )

        dataset = dataset.assemble(
            partial(
                self.build_packing_assembler,
                max_length=max_seq_len,
                selection_strategy=packing.selection_strategy,
                open_pack_limit=packing.open_pack_limit,
                pack_buffer_size=packing.buffer_size,
                defer_finalize=True,
            )
        )

        if resolve_refs:
            dataset = dataset.resolve_ref(ref_names=ref_columns).map(
                partial(self.convert_images_to_pixel_values, processor=processor)
            )

        dataset = dataset.assemble(
            partial(
                self.build_dataguard,
                check_basic_formats=False,
                check_input_ids=True,
                check_image_sizes=False,
                verbose=False,
            )
        )

        dataset = dataset.map(self.finalize_packed_samples)

        return dataset

    def build_collator(
        self, *, pad_token_id: int, processor: Any, ignore_index: int = -100
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Build the standard MLLM collator."""

        class MLLMCollator:
            """Pad and merge preprocessed multimodal samples."""

            DUMMY_IMAGE_SIZE = (32, 32)
            DUMMY_IMAGE_PIXELS = 32 * 32

            def __init__(self, pad_token_id: int, processor: Any, *, ignore_index: int = -100) -> None:
                """Store padding values used during batch collation."""
                self.pad_token_id = pad_token_id
                self.processor = processor
                self.ignore_index = ignore_index
                self._cached_dummy_inputs: dict[str, torch.Tensor] | None = None

            def _get_dummy_image(self) -> Image.Image:
                """Return a cached RGB dummy image reused across text-only batches."""
                cached = getattr(self.processor, "_mvp_basic_vlm_batch_dummy_image", None)
                if isinstance(cached, Image.Image):
                    return cached.copy()

                dummy = Image.new("RGB", self.DUMMY_IMAGE_SIZE, color=0)
                setattr(self.processor, "_mvp_basic_vlm_batch_dummy_image", dummy)
                return dummy.copy()

            def _get_dummy_inputs(self) -> dict[str, torch.Tensor]:
                """Build one valid minimal multimodal suffix for text-only local batches."""
                if self._cached_dummy_inputs is not None:
                    return {key: value.clone() for key, value in self._cached_dummy_inputs.items()}

                fake_messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": self._get_dummy_image()}],
                    }
                ]
                model_inputs = self.processor.apply_chat_template(
                    [fake_messages],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                    min_pixels=self.DUMMY_IMAGE_PIXELS,
                    max_pixels=self.DUMMY_IMAGE_PIXELS,
                )
                self._cached_dummy_inputs = {
                    "input_ids": model_inputs["input_ids"][0].to(dtype=torch.long),
                    "attention_mask": model_inputs["attention_mask"][0].to(dtype=torch.long),
                    "pixel_values": model_inputs["pixel_values"],
                    "image_grid_thw": model_inputs["image_grid_thw"],
                }
                return {key: value.clone() for key, value in self._cached_dummy_inputs.items()}

            def _append_dummy_suffix(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor] | None:
                """Append one active dummy multimodal suffix to sample 0 when the batch is text-only."""
                if any(sample.get("pixel_values") is not None for sample in batch):
                    return None

                dummy_inputs = self._get_dummy_inputs()
                first_sample = batch[0]
                dummy_input_ids = dummy_inputs["input_ids"]

                first_sample["input_ids"] = torch.cat([first_sample["input_ids"], dummy_input_ids], dim=0)
                first_sample["attention_mask"] = torch.cat(
                    [
                        first_sample["attention_mask"],
                        dummy_inputs["attention_mask"].to(first_sample["attention_mask"].dtype),
                    ],
                    dim=0,
                )
                first_sample["labels"] = torch.cat(
                    [
                        first_sample["labels"],
                        torch.full_like(dummy_input_ids, self.ignore_index),
                    ],
                    dim=0,
                )

                next_segment_id = int(first_sample["pack_segment_ids"].max().item()) + 1
                first_sample["pack_segment_ids"] = torch.cat(
                    [
                        first_sample["pack_segment_ids"],
                        torch.full_like(
                            dummy_input_ids,
                            fill_value=next_segment_id,
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )

                return dummy_inputs

            def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
                """Pad token tensors and concatenate optional vision tensors."""
                if not all("pack_segment_ids" in sample for sample in batch):
                    raise ValueError("Packed MLLM samples must include pack_segment_ids.")
                if not all("source_sample_num" in sample for sample in batch):
                    raise ValueError("Packed MLLM samples must include source_sample_num.")

                dummy_inputs = self._append_dummy_suffix(batch)

                model_inputs: dict[str, Any] = {
                    "input_ids": pad_sequence(
                        [sample["input_ids"] for sample in batch],
                        batch_first=True,
                        padding_value=self.pad_token_id,
                    ),
                    "attention_mask": pad_sequence(
                        [sample["attention_mask"] for sample in batch],
                        batch_first=True,
                        padding_value=0,
                    ),
                    "labels": pad_sequence(
                        [sample["labels"] for sample in batch],
                        batch_first=True,
                        padding_value=self.ignore_index,
                    ),
                }

                model_inputs["pack_segment_ids"] = pad_sequence(
                    [sample["pack_segment_ids"] for sample in batch],
                    batch_first=True,
                    padding_value=0,
                )
                model_inputs["source_sample_num"] = torch.tensor(
                    [int(sample["source_sample_num"]) for sample in batch],
                    dtype=torch.long,
                )

                pixel_values = [sample["pixel_values"] for sample in batch if sample.get("pixel_values") is not None]
                if pixel_values:
                    model_inputs["pixel_values"] = torch.cat(pixel_values, dim=0)
                elif dummy_inputs is not None:
                    model_inputs["pixel_values"] = dummy_inputs["pixel_values"]

                image_grid_thw = [
                    sample["image_grid_thw"] for sample in batch if sample.get("image_grid_thw") is not None
                ]
                if image_grid_thw:
                    model_inputs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
                elif dummy_inputs is not None:
                    model_inputs["image_grid_thw"] = dummy_inputs["image_grid_thw"]

                model_inputs["num_input_tokens"] = model_inputs["attention_mask"].sum(dim=-1)
                shifted_labels = torch.nn.functional.pad(model_inputs["labels"], (0, 1), value=self.ignore_index)[
                    ..., 1:
                ]
                model_inputs["num_loss_tokens"] = shifted_labels.ne(self.ignore_index).sum(dim=-1)
                model_inputs["num_source_samples"] = model_inputs["source_sample_num"].clone()
                model_inputs["total_tokens"] = int(model_inputs["num_input_tokens"].sum().item())
                model_inputs["effective_tokens"] = int(model_inputs["num_loss_tokens"].sum().item())

                return model_inputs

        return MLLMCollator(pad_token_id=pad_token_id, processor=processor, ignore_index=ignore_index)

    def build_dataloader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_workers: int,
        collate_fn: Any,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        multiprocessing_context: str = "spawn",
        drop_last: bool = True,
    ):
        """Wrap an mvp_dataset dataset in a TorchLoader batch pipeline."""
        loader = TorchLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )
        return loader.batch(batch_size=int(batch_size), drop_last=drop_last, collate_fn=collate_fn)

    def build_dataguard(
        self,
        assemble_context: RuntimeContext | None = None,
        *,
        check_basic_formats: bool,
        check_input_ids: bool,
        check_image_sizes: bool,
        verbose: bool = True,
    ) -> DataGuard:
        """Build a data guard assembler for an MLLM dataset pipeline."""

        return DataGuard(
            check_basic_formats=check_basic_formats,
            check_input_ids=check_input_ids,
            check_image_sizes=check_image_sizes,
            verbose=verbose,
        )

    def process_sample(
        self,
        sample: dict[str, Any],
        *,
        processor: Any,
        max_length: int,
        image_placeholder: str = IMAGE_PLACEHOLDER,
        ignore_index: int = -100,
        thinking_mode: bool | None | str = True,
    ) -> dict[str, Any]:
        """Process one raw multimodal sample into token/label tensors."""
        return process_sample_impl(
            sample,
            processor=processor,
            max_length=max_length,
            image_placeholder=image_placeholder,
            ignore_index=ignore_index,
            thinking_mode=thinking_mode,
        )

    def convert_images_to_pixel_values(
        self,
        sample: dict[str, Any] | list[dict[str, Any]],
        *,
        processor: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Materialize image references into model pixel tensors."""
        return convert_images_to_pixel_values_impl(sample, processor=processor)

    def build_packing_assembler(
        self,
        assemble_context: RuntimeContext,
        *,
        max_length: int,
        selection_strategy: str,
        open_pack_limit: int,
        pack_buffer_size: int,
        defer_finalize: bool = False,
    ) -> PackingAssembler:
        """Build a packed-sample assembler for an MLLM dataset pipeline."""
        return PackingAssembler(
            max_length=max_length,
            selection_strategy=selection_strategy,
            open_pack_limit=open_pack_limit,
            pack_buffer_size=pack_buffer_size,
            seed=assemble_context.sample_shuffle_seed,
            defer_finalize=defer_finalize,
        )

    def finalize_packed_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Finalize one deferred packed sample group."""
        return finalize_packed_samples(samples)

    def to_device(self, batch: ModelInputs, device: torch.device) -> ModelInputs:
        """Move a batch of token and pixel tensors to the target device."""
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.to(device)
            else:
                batch_on_device[key] = value
        return batch_on_device
