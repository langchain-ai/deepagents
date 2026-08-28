"""Agent middleware covering what MCP tools need around every call.

The tools themselves come out of `langchain.mcp` unmodified. Everything this
app wants on top — an argument quirk, and translating a dead-token failure into
something a model and a user can act on — sits here instead, so the adapter
stays the plain conversion it is and this behavior applies to any MCP tool the
agent is given.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest

logger = logging.getLogger(__name__)

_MCP_MARKER = "_deepagents_code_mcp"
_MCP_SERVER = "_deepagents_code_mcp_server"


def _is_mcp_tool(request: ToolCallRequest) -> bool:
    """Return whether this call is going to a tool adapted from MCP."""
    metadata = getattr(request.tool, "metadata", None) or {}
    return bool(metadata.get(_MCP_MARKER))


def normalize_mcp_arguments(
    arguments: dict[str, Any],
    input_schema: Any,  # noqa: ANN401  # raw JSON Schema dict from the MCP tool
) -> dict[str, Any]:
    """Drop empty-string values for optional MCP tool params.

    Some MCP servers (e.g. Slack's `slack_search_public_and_private`) validate
    optional ID-typed params with `value is not a channel ID` when the model
    fills them in with `""` instead of omitting them. JSON-Schema-derived
    models happily accept `""` for an optional string, so the request reaches
    the server and comes back as a generic failure.

    Treat `""` for a non-required string field as "omitted", so the server sees
    the payload it would have for a field the model genuinely skipped. Required
    fields pass through unchanged, so the server's own missing-field error still
    runs. Only `""` is normalized; `None` is left to the server.

    Args:
        arguments: Arguments the model produced for this call.
        input_schema: The MCP tool's input schema (raw JSON Schema dict).

    Returns:
        A new argument dict, or `arguments` unchanged when nothing was dropped.
    """
    if not isinstance(input_schema, dict):
        return arguments
    required = set(input_schema.get("required") or ())
    properties = input_schema.get("properties") or {}
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if value != "" or key in required:  # noqa: PLC1901  # telling "" from 0/False/[] is the point
            cleaned[key] = value
            continue
        prop = properties.get(key)
        prop_type = prop.get("type") if isinstance(prop, dict) else None
        is_string_typed = prop_type == "string" or (
            isinstance(prop_type, list) and "string" in prop_type
        )
        # Three drop conditions converge here:
        #   - explicit string type (the original Slack-style failure mode);
        #   - missing `type` (oneOf/anyOf/$ref or untyped — treat as ambiguous
        #     and drop, since a server rejects `""` for any ID-shaped slot);
        #   - key absent from `properties` entirely (model invented a field).
        # Anything with an explicit non-string `type` is kept: `""` cannot be a
        # valid integer/bool/array, so it was the model's mistake to send, and
        # the server's own validation says so more clearly than we could.
        if isinstance(prop, dict) and not is_string_typed and prop_type is not None:
            cleaned[key] = value
    if cleaned.keys() != arguments.keys():
        logger.debug(
            "MCP arg normalize: dropped empty-string keys %s",
            sorted(set(arguments) - set(cleaned)),
        )
    return cleaned


def mcp_tool_middleware() -> AgentMiddleware:
    """Build the middleware wrapping every MCP tool call.

    Returns:
        Middleware that normalizes arguments and reports a stale-token failure
            as a tool message naming the login command.
    """

    @wrap_tool_call(name="MCPToolMiddleware")
    async def _wrap(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        tool = request.tool
        if tool is None or not _is_mcp_tool(request):
            return await handler(request)

        server = (tool.metadata or {}).get(_MCP_SERVER, "?")
        arguments = normalize_mcp_arguments(
            request.tool_call.get("args") or {},
            getattr(request.tool, "args_schema", None),
        )
        request = request.override(
            tool_call={**request.tool_call, "args": arguments},
        )

        try:
            return await handler(request)
        # A `ToolException` already reached its tool-local handler, which turned
        # the server's own error content into a failed `ToolMessage`. Re-raising
        # would bury an actionable instruction ("use the X tool instead") under
        # a generic wrapper.
        except ToolException:
            raise
        except Exception as exc:
            from deepagents_code.mcp_auth import find_reauth_required

            reauth = find_reauth_required(exc)
            if reauth is None:
                raise
            # Tokens existed but no longer refresh. The model cannot fix that,
            # so end the call with the instruction the *user* can act on rather
            # than an opaque transport error.
            logger.warning("MCP server %r needs re-authentication: %s", server, reauth)
            return ToolMessage(
                content=str(reauth),
                tool_call_id=request.tool_call["id"],
                name=request.tool_call.get("name"),
                status="error",
            )

    return _wrap


__all__ = ["mcp_tool_middleware", "normalize_mcp_arguments"]
