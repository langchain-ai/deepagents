"""Middleware for renaming tools at the model boundary.

Renames tools in the outbound model request (canonical → alias) and reverts
tool-call names in the inbound response (alias → canonical), keeping all other
middleware and tool routing on canonical Deep Agents names.

Positioning: should be placed late in the middleware stack (after
``_ToolExclusionMiddleware``) so it is the innermost ``wrap_model_call``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage

from deepagents.middleware._tool_exclusion import _tool_name

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
        ResponseT,
    )
    from langchain_core.tools import BaseTool


def _rename_tool(
    tool: BaseTool | dict[str, Any] | object,
    new_name: str,
) -> BaseTool | dict[str, Any] | object:
    """Return a shallow copy of *tool* with its name replaced.

    ``BaseTool`` instances are copied via ``model_copy``; dict tools get a
    shallow ``dict.copy()``; other types (plain callables) are returned
    unchanged.
    """
    if isinstance(tool, dict):
        copied = cast("dict[str, Any]", tool).copy()
        copied["name"] = new_name
        return copied
    if hasattr(tool, "model_copy"):
        return tool.model_copy(update={"name": new_name})
    return tool


def _rewrite_ai_message_tool_calls(
    message: AIMessage,
    aliases: dict[str, str],
) -> AIMessage:
    """Rewrite tool-call names in *message* per *aliases*, returning a copy.

    If no tool calls match the alias map the original message is returned
    (no copy).
    """
    if not message.tool_calls and not message.invalid_tool_calls:
        return message

    changed = False
    rewritten: list[dict[str, Any]] = []
    for tc in message.tool_calls:
        if tc["name"] in aliases:
            rewritten.append({**tc, "name": aliases[tc["name"]]})
            changed = True
        else:
            rewritten.append(tc)

    rewritten_invalid: list[dict[str, Any]] = []
    for tc in message.invalid_tool_calls:
        if tc["name"] in aliases:
            rewritten_invalid.append({**tc, "name": aliases[tc["name"]]})
            changed = True
        else:
            rewritten_invalid.append(tc)

    if not changed:
        return message

    return message.model_copy(
        update={"tool_calls": rewritten, "invalid_tool_calls": rewritten_invalid},
    )


def _rewrite_response_tool_names(
    response: object,  # ModelResponse | AIMessage | ExtendedModelResponse
    aliases: dict[str, str],
) -> object:
    """Rewrite tool-call names in a model response using *aliases*.

    Handles ``AIMessage`` directly and ``ExtendedModelResponse`` (which wraps
    an ``AIMessage`` in its ``.message`` attribute).  Other response types are
    returned unchanged.
    """
    if isinstance(response, AIMessage):
        return _rewrite_ai_message_tool_calls(response, aliases)
    msg = getattr(response, "message", None)
    if isinstance(msg, AIMessage):
        rewritten = _rewrite_ai_message_tool_calls(msg, aliases)
        if rewritten is not msg:
            return response.model_copy(update={"message": rewritten})
    return response


class _ToolAliasingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Rename tools at the model boundary for better model-distribution fit.

    Placed as the innermost ``wrap_model_call`` so all other middleware
    (permissions, HITL, summarization, tool exclusion) continues to see
    canonical Deep Agents tool names.

    Args:
        aliases: Mapping of canonical tool name to model-facing alias
            (e.g. ``{"execute": "shell_command"}``).
    """

    def __init__(self, *, aliases: dict[str, str]) -> None:
        self._forward = aliases
        self._reverse = {v: k for k, v in aliases.items()}

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Rename tools outbound, revert tool-call names inbound."""
        if self._forward:
            renamed = [
                _rename_tool(t, self._forward[name])
                if (name := _tool_name(t)) and name in self._forward
                else t
                for t in request.tools
            ]
            request = request.override(tools=renamed)
        response = handler(request)
        if self._reverse:
            response = _rewrite_response_tool_names(response, self._reverse)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Async variant of ``wrap_model_call``."""
        if self._forward:
            renamed = [
                _rename_tool(t, self._forward[name])
                if (name := _tool_name(t)) and name in self._forward
                else t
                for t in request.tools
            ]
            request = request.override(tools=renamed)
        response = await handler(request)
        if self._reverse:
            response = _rewrite_response_tool_names(response, self._reverse)
        return response
