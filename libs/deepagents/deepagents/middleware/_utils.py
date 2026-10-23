"""Utility functions for middleware."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeGuard

from langchain_core.messages import ContentBlock, SystemMessage

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message.

    Args:
        system_message: Existing system message or None.
        text: Text to add to the system message.

    Returns:
        New SystemMessage with the text appended.
    """
    new_content: list[ContentBlock] = list(system_message.content_blocks) if system_message else []
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=new_content)


def is_anthropic_chat_model(model: object) -> TypeGuard[ChatAnthropic]:
    """Return whether `model` is a `langchain_anthropic` chat model.

    Reads `sys.modules` rather than importing `langchain_anthropic`: an instance
    of `ChatAnthropic` implies its module is loaded, and importing it eagerly
    would add the `anthropic` SDK to every `deepagents` import.
    """
    chat_models = sys.modules.get("langchain_anthropic.chat_models")
    return chat_models is not None and isinstance(model, chat_models.ChatAnthropic)
