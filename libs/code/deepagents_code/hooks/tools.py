"""Native dcode tool vocabulary mapped to compatible wire names."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_code.hooks.models.domain import ToolCallData
    from deepagents_code.json_types import JsonObject

_NATIVE_TO_WIRE_NAME: dict[str, str] = {
    "execute": "Bash",
    "write_file": "Write",
    "edit_file": "Edit",
    "read_file": "Read",
    "glob": "Glob",
    "grep": "Grep",
    "ls": "LS",
}


def to_wire_tool_name(name: str) -> str:
    """Map a native tool name to the compatible wire tool name.

    Args:
        name: Native dcode or already-compatible tool name.

    Returns:
        The Claude-compatible tool name used for matchers and wire payloads.
    """
    if name in _NATIVE_TO_WIRE_NAME:
        return _NATIVE_TO_WIRE_NAME[name]
    return name


def to_wire_tool_input(name: str, args: JsonObject) -> JsonObject:
    """Map native tool arguments to the compatible wire input object.

    Args:
        name: Native dcode tool name (before wire renaming).
        args: Native tool-call arguments.

    Returns:
        JSON object suitable for `tool_input` on the wire.
    """
    del name
    return args


def to_wire_call(call: ToolCallData) -> tuple[str, JsonObject]:
    """Project a native tool call into compatible wire name and input.

    Args:
        call: Native tool-call data from a lifecycle owner.

    Returns:
        `(tool_name, tool_input)` for matchers and wire projection.
    """
    return to_wire_tool_name(call.name), to_wire_tool_input(call.name, call.args)
