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


def _read_input(args: JsonObject) -> JsonObject:
    result = _select(args, "file_path", "limit")
    if "offset" in args:
        offset = args["offset"]
        result["offset"] = (
            offset + 1
            if isinstance(offset, int) and not isinstance(offset, bool)
            else offset
        )
    return result


def _glob_input(args: JsonObject) -> JsonObject:
    return _select(args, "pattern", "path")


def _grep_input(args: JsonObject) -> JsonObject:
    result = _select(args, "path", "glob", "output_mode")
    if "pattern" in args:
        pattern = args["pattern"]
        result["pattern"] = re.escape(pattern) if isinstance(pattern, str) else pattern
    if "max_count" in args:
        result["head_limit"] = args["max_count"]
    return result


def _ls_input(args: JsonObject) -> JsonObject:
    return _select(args, "path")


_NATIVE_TO_WIRE: dict[str, tuple[str, Callable[[JsonObject], JsonObject]]] = {
    "execute": ("Bash", _bash_input),
    "write_file": ("Write", _write_input),
    "edit_file": ("Edit", _edit_input),
    "read_file": ("Read", _read_input),
    "glob": ("Glob", _glob_input),
    "grep": ("Grep", _grep_input),
    "ls": ("LS", _ls_input),
}


def to_wire_tool_name(name: str) -> str:
    """Map a native tool name to the compatible wire tool name.

    Args:
        name: Native dcode or already-compatible tool name.

    Returns:
        The Claude-compatible tool name used for matchers and wire payloads.
    """
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
    return adapter[1](args) if adapter is not None else args


def to_wire_call(call: ToolCallData) -> tuple[str, JsonObject]:
    """Project a native tool call into compatible wire name and input.

    Args:
        call: Native tool-call data from a lifecycle owner.

    Returns:
        `(tool_name, tool_input)` for matchers and wire projection.
    """
    return to_wire_tool_name(call.name), to_wire_tool_input(call.name, call.args)
