"""Normalize reasoning content blocks for `google_genai` model bindings.

`langchain_google_genai._parse_chat_history` reads `content_block["thinking"]`
without a default. Any reasoning block whose text lives under `"text"` /
`"reasoning"`, or is absent entirely, raises `KeyError('thinking')` and crashes
the turn. Coerce those blocks into the shape upstream expects before the
outbound history reaches the provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

from deepagents_code.configurable_model import _get_ls_provider

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


_REASONING_BLOCK_TYPES = frozenset({"reasoning", "thinking"})


def _normalize_reasoning_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Return a canonical `thinking` block, or `None` when it has no text."""
    text = block.get("thinking") or block.get("text") or block.get("reasoning") or ""
    if not text:
        return None
    normalized: dict[str, Any] = {"type": "thinking", "thinking": text}
    signature = block.get("signature")
    if signature is not None:
        normalized["signature"] = signature
    return normalized


def _normalize_content(content: list[Any]) -> list[Any] | None:
    """Return normalized content when any reasoning block was rewritten."""
    changed = False
    new_content: list[Any] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in _REASONING_BLOCK_TYPES:
            normalized = _normalize_reasoning_block(block)
            if normalized is None:
                changed = True
                continue
            if normalized != block:
                changed = True
            new_content.append(normalized)
        else:
            new_content.append(block)
    return new_content if changed else None


def _normalize_messages(messages: list[Any]) -> list[Any] | None:
    """Return a rewritten message list when any AI message needed patching."""
    changed = False
    new_messages: list[Any] = []
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.content, list):
            new_content = _normalize_content(message.content)
            if new_content is not None:
                changed = True
                new_messages.append(message.model_copy(update={"content": new_content}))
                continue
        new_messages.append(message)
    return new_messages if changed else None


def _maybe_patch_request(request: ModelRequest) -> ModelRequest:
    if _get_ls_provider(request.model) != "google_genai":
        return request
    new_messages = _normalize_messages(list(request.messages))
    if new_messages is None:
        return request
    return request.override(messages=new_messages)


class GoogleReasoningFixMiddleware(AgentMiddleware):
    """Rewrite reasoning content blocks so `google_genai` can parse them.

    Workaround for a `KeyError('thinking')` in `langchain_google_genai`'s
    `_parse_chat_history`: it reads `content_block["thinking"]` without a
    default, so blocks that store text under `"text"` / `"reasoning"` or omit
    text entirely crash the turn.
    """

    def wrap_model_call(  # noqa: PLR6301  # AgentMiddleware hook must be an instance method.
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Normalize outbound reasoning blocks for `google_genai` bindings.

        Returns:
            The downstream response from `handler`.
        """
        return handler(_maybe_patch_request(request))

    async def awrap_model_call(  # noqa: PLR6301  # AgentMiddleware hook must be an instance method.
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant of `wrap_model_call`.

        Returns:
            The downstream response from `handler`.
        """
        return await handler(_maybe_patch_request(request))


__all__ = ["GoogleReasoningFixMiddleware"]
