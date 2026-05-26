"""Tokenizer helpers for the Qwen3 LM recipe."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


class ProcessorFingerprint:
    """Pickle-safe callable that returns a stable tokenizer fingerprint."""

    def __init__(self, value: str):
        """Store the precomputed fingerprint string."""
        self.value = value

    def __call__(self) -> str:
        """Return the stored fingerprint."""
        return self.value


class TinyQwenTokenizer:
    """Small deterministic tokenizer for offline smoke tests."""

    def __init__(self, vocab_size: int) -> None:
        """Configure the tiny character tokenizer."""
        self.vocab_size = int(vocab_size)
        self.name_or_path = "tiny-random-qwen3"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.pad_token = "<|pad|>"
        self.eos_token = "<|im_end|>"
        self.padding_side = "right"
        self.__fingerprint__ = ProcessorFingerprint(f"{self.name_or_path}|vocab={self.vocab_size}")

    def __call__(self, text: str, *, add_special_tokens: bool = False, **_: Any) -> dict[str, list[int]]:
        """Tokenize text into deterministic small-vocabulary character ids."""
        input_ids = [self._char_to_id(char) for char in text]
        if add_special_tokens:
            input_ids.append(self.eos_token_id)
        return {"input_ids": input_ids}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
        **_: Any,
    ):
        """Render Qwen-style chat messages for smoke tests."""
        parts: list[str] = []
        for message in messages:
            role = message.get("role")
            content = self._content_to_text(message.get("content", ""))
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if enable_thinking is False:
                parts.append("<think>\n\n</think>\n\n")

        text = "".join(parts)
        if not tokenize:
            return text
        return self(text, add_special_tokens=False)

    def _char_to_id(self, char: str) -> int:
        """Map one Unicode codepoint into the configured tiny vocabulary."""
        return 3 + (ord(char) % max(self.vocab_size - 3, 1))

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Normalize chat content blocks to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
            return "".join(text_parts)
        return str(content)


def build_qwen3_tokenizer(model_config: Any):
    """Load the Qwen3 tokenizer and normalize padding."""
    if bool(getattr(model_config, "tiny_random", False)):
        return TinyQwenTokenizer(int(model_config.tiny_vocab_size))

    tokenizer_path = model_config.tokenizer_name_or_path or model_config.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.__fingerprint__ = ProcessorFingerprint(_tokenizer_fingerprint(tokenizer))
    return tokenizer


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    """Return a stable cache fingerprint for a tokenizer."""
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path:
        return name_or_path
    return f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
