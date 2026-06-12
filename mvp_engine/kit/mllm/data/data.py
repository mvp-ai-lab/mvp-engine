"""Reusable MLLM dataset, dataloader, and collation utilities."""

import re
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from mvp_dataset import Dataset, TorchLoader
from mvp_dataset.core import RuntimeContext
from torch.nn.utils.rnn import pad_sequence

from mvp_engine.utils.log import simple_info

from .guard import DataGuard
from .media import MLLMMediaKit, build_empty_sample
from .packing import PackingAssembler, PackingOptions
from .packing import finalize_packed_samples as finalize_token_packed_samples
from .sample import IMAGE_PLACEHOLDER, MLLMSampleKit
from .step_estimation import StepEstimateResult
from .step_estimation import estimate_total_steps as estimate_packed_total_steps
from .types import ModelInputs

THOUGHT_PREFIX = "<think>\n"
THOUGHT_SUFFIX = "\n</think>\n\n"
THOUGHT_PATTERN = re.compile(f"{re.escape(THOUGHT_PREFIX)}(.*?){re.escape(THOUGHT_SUFFIX)}", re.DOTALL)
THOUGHT_MARKERS = (THOUGHT_PREFIX.strip(), THOUGHT_SUFFIX.strip())
MULTIMODAL_PLACEHOLDER = "<|mvp_multimodal_placeholder|>"


def _resolve_data_parallel_dims(
    device_mesh: object | None,
    dp_dims: str | Sequence[str] | None,
) -> Sequence[str] | str | None:
    """Resolve default data-parallel mesh dimensions for data loading."""
    if dp_dims is not None or device_mesh is None:
        return dp_dims

    mesh_dim_names = tuple(getattr(device_mesh, "mesh_dim_names", ()) or ())
    resolved = tuple(dim_name for dim_name in mesh_dim_names if dim_name != "tensor")
    return resolved or None


class MLLMDataKit:
    """Reusable data utilities for standard MLLM recipes."""

    def __init__(
        self,
        *,
        sample_kit: MLLMSampleKit | None = None,
        media_kit: MLLMMediaKit | None = None,
    ) -> None:
        """Configure sample-schema and media-family behavior for this data pipeline."""
        self.sample_kit = sample_kit or MLLMSampleKit()
        self.media_kit = media_kit or MLLMMediaKit()

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
        dataset_source: str = "lance",
        seed: int = 42,
        packing: PackingOptions = PackingOptions(),
        thinking_mode: bool | None | str = True,
        device_mesh: object | None = None,
        dp_dims: str | Sequence[str] | None = None,
    ) -> Dataset:
        """Build an always-packed MLLM training dataset pipeline."""
        resolved_dp_dims = _resolve_data_parallel_dims(device_mesh, dp_dims)
        if device_mesh is None and resolved_dp_dims is not None:
            raise ValueError("`device_mesh` is required when `dp_dims` is provided.")
        if device_mesh is None or resolved_dp_dims is None:
            context = RuntimeContext.from_runtime(seed=seed)
        else:
            context = RuntimeContext.from_runtime(seed=seed, device_mesh=device_mesh, dp_dims=resolved_dp_dims)

        dataset = Dataset.from_source(
            dataset_source,
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
                partial(self.materialize_media, processor=processor)
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

    def estimate_total_steps(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        gradient_accumulation_steps: int,
        data_parallel_world_size: int,
        data_parallel_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        mode: str = "estimate",
        target_confidence: str = "High",
        confidence_window_size: int = 100,
        sync_interval: int = 10,
    ) -> StepEstimateResult:
        """Infer optimizer steps by consuming a finite packed dataset pipeline."""
        return estimate_packed_total_steps(
            dataset,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            data_parallel_world_size=data_parallel_world_size,
            data_parallel_group=data_parallel_group,
            device=device,
            mode=mode,
            target_confidence=target_confidence,
            confidence_window_size=confidence_window_size,
            sync_interval=sync_interval,
        )

    def build_collator(
        self, *, pad_token_id: int, processor: Any, ignore_index: int = -100
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Build the standard MLLM collator."""

        class MLLMCollator:
            """Pad and merge preprocessed multimodal samples."""

            DUMMY_IMAGE_SIZE = (32, 32)
            DUMMY_IMAGE_PIXELS = 32 * 32

            def __init__(
                self,
                pad_token_id: int,
                processor: Any,
                media_kit: MLLMMediaKit,
                *,
                ignore_index: int = -100,
            ) -> None:
                """Store padding values used during batch collation."""
                self.pad_token_id = pad_token_id
                self.processor = processor
                self.media_kit = media_kit
                self.ignore_index = ignore_index

            def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
                """Pad token tensors and concatenate optional vision tensors."""
                if not all("pack_segment_ids" in sample for sample in batch):
                    raise ValueError("Packed MLLM samples must include pack_segment_ids.")
                if not all("source_sample_num" in sample for sample in batch):
                    raise ValueError("Packed MLLM samples must include source_sample_num.")

                dummy_inputs = self.media_kit._ensure_text_only_batch_has_dummy_media(
                    batch,
                    processor=self.processor,
                    ignore_index=self.ignore_index,
                )

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

                self.media_kit.collate(batch, model_inputs, dummy_inputs=dummy_inputs)

                model_inputs["num_input_tokens"] = model_inputs["attention_mask"].sum(dim=-1)
                shifted_labels = torch.nn.functional.pad(model_inputs["labels"], (0, 1), value=self.ignore_index)[
                    ..., 1:
                ]
                model_inputs["num_loss_tokens"] = shifted_labels.ne(self.ignore_index).sum(dim=-1)
                model_inputs["num_source_samples"] = model_inputs["source_sample_num"].clone()
                model_inputs["total_tokens"] = int(model_inputs["num_input_tokens"].sum().item())
                model_inputs["effective_tokens"] = int(model_inputs["num_loss_tokens"].sum().item())

                return model_inputs

        return MLLMCollator(
            pad_token_id=pad_token_id,
            processor=processor,
            media_kit=self.media_kit,
            ignore_index=ignore_index,
        )

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
        try:
            canonical_sample = self.sample_kit.normalize(sample, image_placeholder=image_placeholder)
            rendered_messages = [dict(message) for message in canonical_sample.messages]
            rendered_messages, assistant_skip_think_prefix = self._apply_thinking_mode(
                rendered_messages,
                thinking_mode=thinking_mode,
            )

            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("Processor does not have a tokenizer attribute for tokenization.")

            media_state = self.media_kit.prepare(
                canonical_sample.media,
                processor=processor,
                tokenizer=tokenizer,
            )
            turns, used_media_count = self._build_chat_sft_turns(
                rendered_messages,
                processor=processor,
                media_state=media_state,
                assistant_skip_think_prefix=assistant_skip_think_prefix,
            )

            input_ids, labels = self._tokenize_chat_sft_turns(
                turns,
                tokenizer=tokenizer,
                media_state=media_state,
                max_length=max_length,
                ignore_index=ignore_index,
            )
            labels = self.media_kit.mask_labels(
                input_ids,
                labels,
                state=media_state,
                ignore_index=ignore_index,
            )
            if not torch.any(labels != ignore_index):
                raise ValueError("has no supervised assistant tokens after tokenization/truncation.")

            processed_sample = dict(sample)
            if media_state.cursor != used_media_count:
                raise ValueError("image size metadata does not match rendered image placeholders.")
            processed_sample.update(media_state.sample_fields)
            processed_sample.update(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": labels,
                }
            )
            return processed_sample
        except Exception as exc:
            simple_info(exc, level="debug")
            return build_empty_sample()

    def materialize_media(
        self,
        sample: dict[str, Any] | list[dict[str, Any]],
        *,
        processor: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Materialize image references into model pixel tensors."""
        return self.media_kit.materialize(sample, processor=processor)

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
        packed_sample = finalize_token_packed_samples(samples)
        self.media_kit._merge_packed(samples, packed_sample)
        return packed_sample

    def to_device(self, batch: ModelInputs, device: torch.device) -> ModelInputs:
        """Move a batch of token and pixel tensors to the target device."""
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.to(device)
            else:
                batch_on_device[key] = value
        return batch_on_device

    def _build_chat_sft_turns(
        self,
        messages: list[dict[str, Any]],
        *,
        processor: Any,
        media_state: Any,
        assistant_skip_think_prefix: set[int],
    ) -> tuple[list[tuple[str, str]], int]:
        """Render canonical chat messages into source/target SFT turns."""
        turns: list[tuple[str, str]] = []
        leading_system_messages: list[dict[str, Any]] = []
        used_media_count = 0
        message_index = 0
        while message_index < len(messages) and messages[message_index].get("role") == "system":
            leading_system_messages.append(messages[message_index])
            message_index += 1

        empty_thought = f"{THOUGHT_PREFIX}{THOUGHT_SUFFIX}"
        while message_index < len(messages):
            user_message = messages[message_index]
            if user_message.get("role") != "user":
                raise ValueError("conversation must contain user/assistant turn pairs after optional system messages.")
            if message_index + 1 >= len(messages):
                break

            assistant_message = messages[message_index + 1]
            if assistant_message.get("role") != "assistant":
                raise ValueError("conversation must contain user/assistant turn pairs after optional system messages.")

            source_messages = leading_system_messages + [user_message]
            full_messages = source_messages + [assistant_message]
            used_media_count += self._count_message_media_blocks(full_messages)
            raw_source_text = processor.apply_chat_template(
                source_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            raw_full_text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not raw_full_text.startswith(raw_source_text):
                raise ValueError("processor chat template does not preserve source prefix for assistant target split.")

            source_text = self.media_kit.render_text(raw_source_text, media_state)
            target_text = self.media_kit.render_text(raw_full_text[len(raw_source_text) :], media_state)
            if message_index + 1 in assistant_skip_think_prefix and target_text.startswith(empty_thought):
                source_text += empty_thought
                target_text = target_text[len(empty_thought) :]

            turns.append((source_text, target_text))
            leading_system_messages = []
            message_index += 2

        return turns, used_media_count

    def _tokenize_chat_sft_turns(
        self,
        turns: list[tuple[str, str]],
        *,
        tokenizer: Any,
        media_state: Any,
        max_length: int,
        ignore_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize source/target turns and build assistant-only labels."""
        input_ids_list: list[int] = []
        labels_list: list[int] = []
        total_length = 0
        for source_text, target_text in turns:
            if total_length >= max_length:
                break

            source_ids = tokenizer(source_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
            source_len, target_len = self._infer_seqlen(
                len(source_ids),
                len(target_ids),
                max_length - total_length,
            )
            self.media_kit.check_truncation(source_ids, source_len, state=media_state)
            self.media_kit.check_truncation(target_ids, target_len, state=media_state)

            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            input_ids_list.extend(source_ids)
            input_ids_list.extend(target_ids)
            labels_list.extend([ignore_index] * len(source_ids))
            labels_list.extend(target_ids)
            total_length += len(source_ids) + len(target_ids)

        return torch.tensor(input_ids_list, dtype=torch.long), torch.tensor(labels_list, dtype=torch.long)

    def _apply_thinking_mode(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking_mode: bool | None | str,
    ) -> tuple[list[dict[str, Any]], set[int]]:
        """Rewrite assistant thinking blocks before tokenization."""
        rewritten_messages: list[dict[str, Any]] = []
        assistant_skip_think_prefix: set[int] = set()

        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                rewritten_messages.append(message)
                continue

            content = message.get("content")
            non_text_blocks: list[dict[str, Any]] = []
            preserve_blocks = isinstance(content, list)

            if isinstance(content, str):
                content_text = content
            elif isinstance(content, list):
                text_parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                        text_parts.append(block["text"])
                    elif isinstance(block, dict):
                        text_parts.append(MULTIMODAL_PLACEHOLDER)
                        non_text_blocks.append(block)
                content_text = "".join(text_parts)
            else:
                content_text = ""

            thought_match = THOUGHT_PATTERN.search(content_text)
            thought_is_empty = thought_match is None or not thought_match.group(1).strip()
            modified_text = content_text

            if thinking_mode is False:
                modified_text = THOUGHT_PATTERN.sub("", content_text).lstrip("\n")
            elif thinking_mode == "non-empty" and thought_is_empty:
                modified_text = THOUGHT_PATTERN.sub("", content_text).lstrip("\n")

            has_thought_block = all(marker in modified_text for marker in THOUGHT_MARKERS)
            should_add_empty_thought = False
            should_skip_added_thought = False

            if not has_thought_block:
                if thinking_mode is False:
                    should_add_empty_thought = True
                    should_skip_added_thought = True
                elif thinking_mode == "non-empty":
                    should_add_empty_thought = thought_is_empty
                    should_skip_added_thought = thought_is_empty
                elif thinking_mode is not None:
                    should_add_empty_thought = True

            if should_add_empty_thought:
                modified_text = f"{THOUGHT_PREFIX}{THOUGHT_SUFFIX}{modified_text}"
                if should_skip_added_thought:
                    assistant_skip_think_prefix.add(message_index)

            rewritten_message = dict(message)
            if preserve_blocks:
                if non_text_blocks:
                    parts = modified_text.split(MULTIMODAL_PLACEHOLDER)
                    rebuilt_blocks: list[dict[str, Any]] = []
                    for part_index, part in enumerate(parts):
                        if part:
                            rebuilt_blocks.append({"type": "text", "text": part})
                        if part_index < len(non_text_blocks):
                            rebuilt_blocks.append(non_text_blocks[part_index])
                    rewritten_message["content"] = rebuilt_blocks
                else:
                    rewritten_message["content"] = [{"type": "text", "text": modified_text}] if modified_text else []
            else:
                rewritten_message["content"] = modified_text

            rewritten_messages.append(rewritten_message)

        return rewritten_messages, assistant_skip_think_prefix

    @staticmethod
    def _count_message_media_blocks(messages: list[dict[str, Any]]) -> int:
        """Count media blocks rendered into one source/target turn."""
        return sum(
            1
            for message in messages
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") != "text"
        )

    @staticmethod
    def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
        """Allocate the remaining token budget between source and target text."""
        if target_len * 2 < cutoff_len:
            max_target_len = cutoff_len
        elif source_len * 2 < cutoff_len:
            max_target_len = cutoff_len - source_len
        else:
            max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

        new_target_len = min(max_target_len, target_len)
        max_source_len = max(cutoff_len - new_target_len, 0)
        return min(max_source_len, source_len), new_target_len
