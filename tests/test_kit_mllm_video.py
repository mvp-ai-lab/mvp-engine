"""Unit tests for the generic video media kit (``mvp_engine.kit.mllm.data.video``)."""

from __future__ import annotations

import pytest
import torch

from mvp_engine.kit.mllm.data.video import VideoMediaKit


class _FakeTokenizer:
    """Whitespace tokenizer mapping the video token to a fixed id and others to stable ids."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _id(self, token: str) -> int:
        return self._vocab.setdefault(token, 1000 + len(self._vocab))

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        tokens = text.replace("<vid>", " <vid> ").split()
        return {"input_ids": [999 if tok == "<vid>" else self._id(tok) for tok in tokens]}


class _FakeProcessor:
    """Minimal processor whose chat template keeps the prompt prefix and one video token."""

    video_token = "<vid>"
    video_token_id = 999

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False) -> str:
        rendered = []
        for message in messages:
            blocks = [b["text"] if b["type"] == "text" else "<vid>" for b in message["content"]]
            rendered.append(f"<|{message['role']}|> " + " ".join(blocks))
        text = " ".join(rendered)
        if add_generation_prompt:
            text += " <|assistant|>"
        return text


def _sample(messages: list[dict]) -> dict:
    return {"conversations": messages}


def test_render_chat_single_video_splits_for_supervision():
    prompt, target = VideoMediaKit().render_chat(
        _sample(
            [
                {"from": "human", "value": "<video>\nwhat happens?"},
                {"from": "gpt", "value": "a cat jumps."},
            ]
        )
    )
    assert [m["role"] for m in prompt] == ["user"]
    assert [m["role"] for m in target] == ["user", "assistant"]
    user_blocks = target[0]["content"]
    assert {"type": "video"} in user_blocks
    assert any(b["type"] == "text" for b in user_blocks)


def test_render_chat_prepends_video_for_caption_rows():
    _, target = VideoMediaKit().render_chat(
        _sample(
            [
                {"from": "human", "value": "describe the clip."},
                {"from": "gpt", "value": "a dog runs."},
            ]
        )
    )
    assert target[0]["content"][0] == {"type": "video"}


def test_render_chat_rejects_multiple_videos():
    with pytest.raises(ValueError):
        VideoMediaKit().render_chat(
            _sample([{"from": "human", "value": "<video><video> two?"}, {"from": "gpt", "value": "no"}])
        )


def test_render_chat_requires_video_before_assistant():
    with pytest.raises(ValueError):
        VideoMediaKit().render_chat(
            _sample([{"from": "human", "value": "hi"}, {"from": "gpt", "value": "<video> answer"}])
        )


def test_render_chat_requires_assistant_turn():
    with pytest.raises(ValueError):
        VideoMediaKit().render_chat(_sample([{"from": "human", "value": "<video> only"}]))


def test_build_inputs_and_labels_expands_video_and_masks_prompt():
    kit = VideoMediaKit()
    prompt, target = kit.render_chat(
        _sample([{"from": "human", "value": "<video> q"}, {"from": "gpt", "value": "ans"}])
    )
    input_ids, attention_mask, labels = kit.build_inputs_and_labels(
        prompt_messages=prompt,
        target_messages=target,
        processor=_FakeProcessor(),
        video_token_count=3,
        max_length=128,
    )
    assert int((input_ids == 999).sum()) == 3  # the single placeholder expands to 3 pad tokens
    assert torch.equal(attention_mask, torch.ones_like(input_ids))
    assert int((labels == 999).sum()) == 0  # video tokens are masked from the loss
    supervised = labels[labels != -100]
    assert supervised.numel() >= 1  # the assistant answer is supervised
    assert torch.equal(supervised, input_ids[labels != -100])


def test_build_generation_inputs_prompt_only_with_video_expansion():
    kit = VideoMediaKit()
    prompt, _ = kit.render_chat(
        _sample([{"from": "human", "value": "<video> q"}, {"from": "gpt", "value": "ans"}])
    )
    ids = kit.build_generation_inputs(prompt_messages=prompt, processor=_FakeProcessor(), video_token_count=3)
    assert ids.dim() == 1 and ids.numel() > 3
    assert int((ids == 999).sum()) == 3  # single <video> expanded to 3 pad tokens, same as training
    assert 999 not in ids[-2:].tolist()  # prompt ends with the generation-prompt tail, not a video token


def test_build_inputs_and_labels_rejects_overlength():
    kit = VideoMediaKit()
    prompt, target = kit.render_chat(
        _sample([{"from": "human", "value": "<video> q"}, {"from": "gpt", "value": "ans"}])
    )
    with pytest.raises(ValueError):
        kit.build_inputs_and_labels(
            prompt_messages=prompt,
            target_messages=target,
            processor=_FakeProcessor(),
            video_token_count=3,
            max_length=2,
        )
