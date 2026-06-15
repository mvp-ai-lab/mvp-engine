"""Reusable text-LM pretraining dataset, packing, and collation utilities."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from mvp_dataset import Dataset, TorchLoader
from mvp_dataset.core import Assembler, RuntimeContext
from torch.nn.utils.rnn import pad_sequence

from .guard import DataGuard
from .packing import PackingAssembler, PackingOptions
from .packing import finalize_packed_samples as finalize_token_packed_samples
from .types import ModelInputs

IGNORE_INDEX = -100


class TokenizeAssembler(Assembler):
    """Tokenize one raw text row and split overlong documents into chunks.

    Pretraining trains on every token, so each chunk's ``labels`` are just a copy
    of ``input_ids``. Cross-document boundary masking happens later in
    ``LLMDataKit.finalize_packed_samples``.
    """

    def __init__(self, tokenizer: Any, max_length: int, text_field: str):
        """Store the tokenizer, per-chunk length cap, and raw text field name."""
        super().__init__()
        if int(max_length) <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}.")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.text_field = text_field

    def push(self, sample: dict[str, Any]):
        """Tokenize ``sample[text_field]`` and emit one or more chunk samples."""
        text = sample.get(self.text_field)
        if not isinstance(text, str) or not text.strip():
            return []

        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id; required to mark document ends.")
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = ids + [self.tokenizer.eos_token_id]  # EOS marks the document end

        chunks = []
        for start in range(0, len(ids), self.max_length):
            piece = ids[start : start + self.max_length]
            input_ids = torch.tensor(piece, dtype=torch.long)
            chunks.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": input_ids.clone(),  # full-token loss
                }
            )
        return chunks

    def finish(self, *, drop_last: bool = False):
        """No buffered state to flush."""
        del drop_last
        return []


class LLMCollator:
    """Pad packed text samples and count tokens for the token-normalized loss."""

    def __init__(self, pad_token_id: int):
        """Store the pad token id used during batch collation."""
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> ModelInputs:
        """Pad token tensors to the longest sample and compute token counts."""
        input_ids = pad_sequence(
            [sample["input_ids"] for sample in batch], batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = pad_sequence([sample["attention_mask"] for sample in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([sample["labels"] for sample in batch], batch_first=True, padding_value=IGNORE_INDEX)
        pack_segment_ids = pad_sequence(
            [sample["pack_segment_ids"] for sample in batch], batch_first=True, padding_value=0
        )
        source_sample_num = torch.tensor([int(sample["source_sample_num"]) for sample in batch], dtype=torch.long)

        num_input_tokens = attention_mask.sum(dim=-1)
        shifted_labels = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
        num_loss_tokens = shifted_labels.ne(IGNORE_INDEX).sum(dim=-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pack_segment_ids": pack_segment_ids,
            "source_sample_num": source_sample_num,
            "num_input_tokens": num_input_tokens,
            "num_loss_tokens": num_loss_tokens,
            "num_source_samples": source_sample_num.clone(),
            "total_tokens": int(num_input_tokens.sum().item()),
            "effective_tokens": int(num_loss_tokens.sum().item()),
        }


class LLMDataKit:
    """Reusable data utilities for text-only LM pretraining."""

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
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )
        tokenizer.padding_side = padding_side
        if pad_token_fallback_to_eos and tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def build_dataset(
        self,
        dataset_path: str,
        *,
        tokenizer: Any,
        max_seq_len: int,
        text_field: str = "data",
        resample: bool = True,
        dataset_source: str = "lance",
        seed: int = 42,
        packing: PackingOptions = PackingOptions(),
    ) -> Dataset:
        """Build the always-packed text pretraining dataset pipeline."""
        context = RuntimeContext.from_runtime(seed=seed)

        dataset = Dataset.from_source(
            dataset_source,
            dataset_path,
            context=context,
            resample=resample,
            shuffle_mode="fragment_aware",
        )
        dataset = dataset.assemble(
            partial(self.build_dataguard, check_basic_formats=True, check_input_ids=False, text_field=text_field)
        )
        dataset = dataset.assemble(
            partial(
                self.build_tokenize_assembler, tokenizer=tokenizer, max_length=int(max_seq_len), text_field=text_field
            )
        )
        dataset = dataset.assemble(
            partial(self.build_dataguard, check_basic_formats=False, check_input_ids=True, text_field=text_field)
        )
        dataset = dataset.assemble(
            partial(
                self.build_packing_assembler,
                max_length=int(max_seq_len),
                selection_strategy=packing.selection_strategy,
                open_pack_limit=packing.open_pack_limit,
                pack_buffer_size=packing.buffer_size,
                defer_finalize=True,
            )
        )
        dataset = dataset.map(self.finalize_packed_samples)
        return dataset

    def build_collator(self, *, pad_token_id: int) -> Callable[[list], ModelInputs]:
        """Build the text collator that pads tensors and counts tokens."""
        return LLMCollator(pad_token_id=pad_token_id)

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
        text_field: str = "data",
        verbose: bool = True,
    ) -> DataGuard:
        """Build a text data guard for one stage of the dataset pipeline."""
        return DataGuard(
            check_basic_formats=check_basic_formats,
            check_input_ids=check_input_ids,
            text_field=text_field,
            verbose=verbose,
        )

    def build_tokenize_assembler(
        self,
        assemble_context: RuntimeContext,
        *,
        tokenizer: Any,
        max_length: int,
        text_field: str,
    ) -> TokenizeAssembler:
        """Build the tokenize-and-chunk assembler for the dataset pipeline."""
        return TokenizeAssembler(tokenizer=tokenizer, max_length=max_length, text_field=text_field)

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
        """Build a packed-sample assembler for the dataset pipeline."""
        return PackingAssembler(
            max_length=max_length,
            selection_strategy=selection_strategy,
            open_pack_limit=open_pack_limit,
            pack_buffer_size=pack_buffer_size,
            seed=assemble_context.sample_shuffle_seed,
            defer_finalize=defer_finalize,
        )

    def finalize_packed_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate one packed group and mask loss at each document boundary.

        ``apply_chunked_token_loss_patch`` shifts labels by one, so position ``i``
        is trained to predict ``labels[i + 1]``. To stop the last token of one
        document from predicting the first token of the next (which the block mask
        forbids it to attend to), we set the label at every segment start to
        ``IGNORE_INDEX``.
        """
        packed = finalize_token_packed_samples(samples)
        segment_ids = packed["pack_segment_ids"]
        labels = packed["labels"]

        is_segment_start = torch.ones_like(segment_ids, dtype=torch.bool)
        is_segment_start[1:] = segment_ids[1:] != segment_ids[:-1]
        labels[is_segment_start] = IGNORE_INDEX

        packed["labels"] = labels
        return packed

    def to_device(self, batch: ModelInputs, device: torch.device) -> ModelInputs:
        """Move all tensor fields of a batch to the target device."""
        batch_on_device: dict[str, Any] = {}
        for key, value in batch.items():
            batch_on_device[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return batch_on_device
