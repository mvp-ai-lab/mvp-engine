"""Qwen text-chat tokenization aliases."""

from __future__ import annotations

from dataclasses import dataclass

from ..tokenization import LLMTokenizationHandler


@dataclass(slots=True)
class QwenChatTokenizationHandler(LLMTokenizationHandler):
    """Tokenization handler for Qwen chat-template segments."""

    add_eos: bool = False


__all__ = ["QwenChatTokenizationHandler"]
