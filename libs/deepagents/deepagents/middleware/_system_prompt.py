"""SystemPromptAssemblerMiddleware — final assembly of named system-prompt sections."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from deepagents.middleware._utils import (
    _BASE_SYSTEM_MESSAGE_KEY,
    _SYSTEM_SECTIONS_KEY,
    assemble_system_message,
)


class SystemPromptAssemblerMiddleware(AgentMiddleware[Any, Any, Any]):
    """Innermost tail middleware that finalises the system prompt.

    Middleware earlier in the stack contribute named sections via
    :func:`~deepagents.middleware._utils.set_system_section`.  This middleware
    reads those sections from ``request.model_settings``, assembles them into
    the final ``SystemMessage`` in deterministic order, strips the internal
    ``_deepagents_system_sections`` key so it never reaches the LLM API, and
    calls the next handler with the clean request.

    Assembly order
    --------------
    1. Base ``system_message`` blocks (user-supplied prompt, if any).
    2. Named sections sorted by ``(section.order, section.key)``.

    ``cache_control`` ephemeral breakpoints are applied to any section where
    ``cache_control=True`` when the active model is an Anthropic model.
    """

    def _assemble(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        sections: dict[str, Any] | None = request.model_settings.get(_SYSTEM_SECTIONS_KEY)
        if not sections:
            return request
        # Rebuild from the original base (saved before any set_system_section calls)
        # so sections are ordered correctly instead of in registration order.
        base = request.model_settings.get(_BASE_SYSTEM_MESSAGE_KEY)
        clean_settings = {
            k: v
            for k, v in request.model_settings.items()
            if k not in (_SYSTEM_SECTIONS_KEY, _BASE_SYSTEM_MESSAGE_KEY)
        }
        assembled = assemble_system_message(base, sections, model=request.model)
        return request.override(system_message=assembled, model_settings=clean_settings)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage:
        return handler(self._assemble(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        return await handler(self._assemble(request))
