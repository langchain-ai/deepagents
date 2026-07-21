"""Native dcode tool vocabulary mapped to compatible wire names."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents_code.hooks.models.domain import ToolCallData
    from deepagents_code.json_types import JsonObject


def _select(args: JsonObject, *fields: str) -> JsonObject:
    return {field: args[field] for field in fields if field in args}


def _bash_input(args: JsonObject) -> JsonObject:
    result = _select(args, "command")
    if "timeout" in args:
        timeout = args["timeout"]
        result["timeout"] = (
            timeout * 1000
            if isinstance(timeout, int) and not isinstance(timeout, bool)
            else timeout
        )
    return result


def _write_input(args: JsonObject) -> JsonObject:
    return _select(args, "file_path", "content")


def _edit_input(args: JsonObject) -> JsonObject:
    return _select(args, "file_path", "old_string", "new_string", "replace_all")


_NATIVE_TO_WIRE: dict[str, tuple[str, Callable[[JsonObject], JsonObject]]] = {
    "execute": ("Bash", _bash_input),
    "write_file": ("Write", _write_input),
    "edit_file": ("Edit", _edit_input),
}

_MCP_WIRE_RE = re.compile(r"^mcp__.+__.+$")


def format_mcp_wire_name(server: str, tool: str) -> str:
    """Format a compatible MCP wire tool name.

    Args:
        server: MCP server name.
        tool: Bare MCP tool name.

    Returns:
        `mcp__{server}__{tool}`.
    """
    return f"mcp__{server}__{tool}"


def to_wire_tool_name(
    name: str,
    *,
    mcp_server: str | None = None,
) -> str:
    """Map a native tool name to the compatible wire tool name.

    Args:
        name: Native dcode or already-compatible tool name.
        mcp_server: Owning MCP server when known from tool metadata.

    Returns:
        The Claude-compatible tool name used for matchers and wire payloads.
    """
    if _MCP_WIRE_RE.fullmatch(name) is not None:
        return name
    if mcp_server is not None:
        prefix = f"{mcp_server}_"
        tool = name.removeprefix(prefix) if name.startswith(prefix) else name
        return format_mcp_wire_name(mcp_server, tool)
    adapter = _NATIVE_TO_WIRE.get(name)
    return adapter[0] if adapter is not None else name


def to_wire_tool_input(name: str, args: JsonObject) -> JsonObject:
    """Map native tool arguments to the compatible wire input object.

    Args:
        name: Native dcode tool name (before wire renaming).
        args: Native tool-call arguments.

    Returns:
        JSON object suitable for `tool_input` on the wire.
    """
    adapter = _NATIVE_TO_WIRE.get(name)
    return adapter[1](args) if adapter is not None else dict(args)


def to_wire_call(
    call: ToolCallData,
) -> tuple[str, JsonObject]:
    """Project a native tool call into compatible wire name and input.

    Args:
        call: Native tool-call data from a lifecycle owner.

    Returns:
        `(tool_name, tool_input)` for matchers and wire projection.
    """
    name = to_wire_tool_name(call.name, mcp_server=call.mcp_server)
    if call.mcp_server is not None or _MCP_WIRE_RE.fullmatch(call.name) is not None:
        return name, dict(call.args)
    return name, to_wire_tool_input(call.name, call.args)
