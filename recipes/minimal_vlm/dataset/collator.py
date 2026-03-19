"""Batch collation for the minimal VLM recipe."""

from __future__ import annotations

from typing import Any

import torch


class MinimalVlmCollator:
    """Collate conversation samples with a Hugging Face multimodal processor."""

    def __init__(self, processor: Any, max_length: int, *, ignore_index: int = -100) -> None:
        self.processor = processor
        self.max_length = max_length
        self.ignore_index = ignore_index

    def _tokenize_conversation(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> torch.Tensor:
        tokenized = self.processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        return tokenized["input_ids"][0]

    def _build_labels_for_sample(
        self,
        messages: list[dict[str, Any]],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise assistant response tokens by comparing message prefixes."""
        labels = torch.full_like(input_ids, self.ignore_index)

        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            prefix_length = 0
            if message_index > 0:
                prefix_ids = self._tokenize_conversation(
                    messages[:message_index],
                    add_generation_prompt=True,
                )
                prefix_length = int(prefix_ids.size(0))

            upto_ids = self._tokenize_conversation(
                messages[: message_index + 1],
                add_generation_prompt=False,
            )
            upto_length = int(upto_ids.size(0))

            start = min(prefix_length, input_ids.size(0))
            end = min(upto_length, input_ids.size(0))
            if start < end:
                labels[start:end] = input_ids[start:end]

        return labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        conversations = [sample["messages"] for sample in batch]
        model_inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = torch.full_like(input_ids, self.ignore_index)

        for batch_index, sample in enumerate(batch):
            valid_length = int(attention_mask[batch_index].sum().item())
            if valid_length <= 0:
                continue

            labels[batch_index, :valid_length] = self._build_labels_for_sample(
                sample["messages"],
                input_ids[batch_index, :valid_length],
            )

        model_inputs["labels"] = labels

        return model_inputs
