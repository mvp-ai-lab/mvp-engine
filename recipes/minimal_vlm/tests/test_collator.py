from __future__ import annotations

import torch

from recipes.minimal_vlm.dataset import MinimalVlmCollator


class DummyProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
        if len(conversations) == 2:
            return {
                "input_ids": torch.tensor(
                    [
                        [10, 11, 12, 13, 14, 15, 16, 0],
                        [20, 21, 22, 0, 0, 0, 0, 0],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0],
                    ]
                ),
                "pixel_values": torch.randn(2, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            }

        messages = conversations[0]
        message_count = len(messages)
        add_generation_prompt = kwargs["add_generation_prompt"]

        if message_count == 1 and add_generation_prompt:
            return {"input_ids": torch.tensor([[20, 21]])}
        if message_count == 1 and not add_generation_prompt:
            return {"input_ids": torch.tensor([[20, 21, 22]])}
        if message_count == 2 and not add_generation_prompt:
            return {"input_ids": torch.tensor([[10, 11, 12, 13]])}
        if message_count == 3 and add_generation_prompt:
            return {"input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]])}
        if message_count == 4 and not add_generation_prompt:
            return {"input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]])}

        raise AssertionError(
            f"Unexpected call: message_count={message_count}, add_generation_prompt={add_generation_prompt}"
        )


def test_collator_masks_non_assistant_and_padding_tokens() -> None:
    processor = DummyProcessor()
    collator = MinimalVlmCollator(processor=processor, max_length=16)
    batch = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "u1"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
                {"role": "user", "content": [{"type": "text", "text": "u2"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
            ]
        },
        {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "a"}]}]},
    ]

    model_inputs = collator(batch)

    assert torch.equal(
        model_inputs["labels"],
        torch.tensor(
            [
                [-100, -100, 12, 13, -100, -100, 16, -100],
                [20, 21, 22, -100, -100, -100, -100, -100],
            ]
        ),
    )

    _, kwargs = processor.calls[0]
    assert kwargs["max_length"] == 16
