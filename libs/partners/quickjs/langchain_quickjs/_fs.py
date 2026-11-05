"""Node-shaped `fs` over the `ptc` file tools: one `_OPERATIONS` row per tool call."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

from deepagents.backends import protocol as _protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool

    from langchain_quickjs._dispatch import CallResult

MAX_READ_BYTES = 8 * 1024 * 1024
"""Largest file one read copies into the REPL heap, the journal, and traces."""

# Older deepagents releases lack the constant; the value is the same. Remove the
# fallback in the chore(deps) PR that raises the `deepagents` floor past them.
RAW_FILE_RESULT_CONFIG_KEY = getattr(
    _protocol, "RAW_FILE_RESULT_CONFIG_KEY", "deepagents_raw_file_result"
)

SYNC_HOST_FUNCTION = "__lc_fs_sync"
ASYNC_HOST_FUNCTION = "__lc_fs_async"


@dataclass(frozen=True, slots=True)
class _Operation:
    tool: str
    syscall: str
    """The syscall Node names in this operation's errors."""
    arguments: Callable[[dict[str, Any]], dict[str, Any]]
    value: Callable[[dict[str, Any], Any], Any]
    """Node's arguments and the tool's raw result to the value the shim gets."""
    doc: str


_DECODERS: dict[str | None, Callable[[bytes], str | bytes]] = {
    None: lambda b: b,
    "utf8": lambda b: b.decode("utf-8", "replace"),
    "utf-8": lambda b: b.decode("utf-8", "replace"),
    "base64": lambda b: base64.b64encode(b).decode("ascii"),
}


_OPERATIONS = {
    "readFile": _Operation(
        tool="read_file",
        syscall="open",
        arguments=lambda a: {"file_path": a["path"]},
        value=lambda a, raw: _DECODERS[a["encoding"]](raw["content"]),
        doc=(
            '- `fs.readFileSync(path, "utf8")` returns the whole file as a string, '
            "with no line numbers or headers, as in "
            '`JSON.parse(fs.readFileSync("/config.json", "utf8"))`. Without an '
            "encoding it returns a `Uint8Array` whose `toString()` decodes UTF-8; "
            '`"base64"` also works.'
        ),
    ),
}

_NODE_ERRORS = {
    "file_not_found": ("ENOENT", "no such file or directory"),
    "is_directory": ("EISDIR", "illegal operation on a directory"),
    "permission_denied": ("EACCES", "permission denied"),
    "invalid_path": ("EINVAL", "invalid argument"),
    "file_too_large": ("ERR_FS_FILE_TOO_LARGE", "file is over the 8 MiB read limit"),
}
"""Node's code and message for each code a file tool reports; others are `EIO`."""


def docs(tool_names: Collection[str]) -> list[str]:
    """Return the prompt lines for the `fs` functions whose tool is listed."""
    return [op.doc for op in _OPERATIONS.values() if op.tool in tool_names]


def request_raw_result(config: RunnableConfig, tool_call_id: str) -> RunnableConfig:
    """Copy `config` so the deepagents file tool call `tool_call_id` is raw."""
    raw = {"tool_call_id": tool_call_id, "max_bytes": MAX_READ_BYTES}
    configurable = {**config.get("configurable", {}), RAW_FILE_RESULT_CONFIG_KEY: raw}
    return {**config, "configurable": configurable}


def request(
    op: str, args: dict[str, Any], tools: Mapping[str, BaseTool]
) -> tuple[BaseTool, dict[str, Any]] | list[Any]:
    """Return the tool and arguments for one `fs` call, or the error to throw."""
    operation = _OPERATIONS[op]
    if args.get("encoding") not in _DECODERS:
        msg = f"The argument 'encoding' is invalid. Received {args['encoding']!r}"
        return [1, "ERR_INVALID_ARG_VALUE", msg, {"name": "TypeError"}]
    tool = tools.get(operation.tool)
    if tool is None:
        detail = f"function not implemented; add {operation.tool!r} to `ptc`"
        return _error("ENOSYS", detail, op, args)
    return tool, operation.arguments(args)


def response(op: str, args: dict[str, Any], result: CallResult) -> list[Any]:
    """Turn a file tool's call result into the envelope the JS shim settles."""
    operation = _OPERATIONS[op]
    if result.status == "error":
        return _error(*_failure(result), op, args)
    try:
        return [0, operation.value(args, result.artifact)]
    except (KeyError, TypeError):
        if result.artifact is None:
            detail = f"the {operation.tool} tool returned no raw result for fs"
            return _error("ENOSYS", detail, op, args)
        detail = f"the {operation.tool} tool returned a raw result fs cannot read"
        return _error("EIO", detail, op, args)


def _failure(result: CallResult) -> tuple[str, str]:
    """Map a failed call to a Node code and detail; a rejection is `EACCES`."""
    text = str(result.value).removeprefix("Error: ")
    if result.rejected:
        return "EACCES", text
    artifact = result.artifact if isinstance(result.artifact, dict) else {}
    return _NODE_ERRORS.get(artifact.get("code"), ("EIO", text))


def _error(code: str, detail: str, op: str, args: dict[str, Any]) -> list[Any]:
    syscall = _OPERATIONS[op].syscall
    path = args.get("path", "/")
    message = f"{code}: {detail}, {syscall} '{path}'"
    return [1, code, message, {"syscall": syscall, "path": path}]


SHIM = resources.files("langchain_quickjs").joinpath("_fs_shim.js").read_text("utf-8")
"""A JS function expression that installs `fs`; see `_fs_shim.js`."""
