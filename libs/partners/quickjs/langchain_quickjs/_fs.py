"""Read-only, Node-shaped `fs` for the REPL, over the agent's own file tools.

Each operation calls the agent's `read_file` or `ls` tool through the
`tools.*` bridge: path checks and deny rules apply, approval gates do not.
A request on that one call's runtime config asks the tool for its raw
result on `ToolMessage.artifact` instead of the text formatted for the model.
Host exceptions reach JS as sanitized `HostError`s, which can't carry a `code`,
so expected failures come back as `{__lc_fs_error: {...}}` values that the JS
shim throws as Node errors.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, replace
from importlib import resources
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

_DEFAULT_MAX_READ_BYTES = 8 * 1024 * 1024

_RAW_RESULT_CONFIG_KEY = "deepagents_raw_file_result"
"""deepagents' `RAW_FILE_RESULT_CONFIG_KEY`, spelled out so this module imports
on releases that predate it. Their file tools ignore it, and `fs` throws `ENOSYS`.
"""

SYNC_HOST_FUNCTION = "__lc_fs_sync"
ASYNC_HOST_FUNCTION = "__lc_fs_async"
"""Host functions the shim calls by global name, so a restored snapshot finds them."""


class _Operation(NamedTuple):
    tool: str
    syscall: str
    """The syscall Node names in this operation's errors."""
    arguments: Callable[[dict[str, Any]], dict[str, Any]]


_OPERATIONS = {
    "readFile": _Operation("read_file", "open", lambda a: {"file_path": a["path"]}),
    "readdir": _Operation("ls", "scandir", lambda a: {"path": a["path"]}),
}

TOOL_NAMES = frozenset(operation.tool for operation in _OPERATIONS.values())
"""The agent's tools that `fs` reads through."""

_DECODERS: dict[str | None, Callable[[bytes], str | bytes]] = {
    None: lambda b: b,
    "utf8": lambda b: b.decode("utf-8", "replace"),
    "utf-8": lambda b: b.decode("utf-8", "replace"),
}

_NODE_ERRORS = {
    "file_not_found": ("ENOENT", "no such file or directory"),
    "is_directory": ("EISDIR", "illegal operation on a directory"),
    "permission_denied": ("EACCES", "permission denied"),
    "invalid_path": ("EINVAL", "invalid argument"),
    "file_too_large": ("ERR_FS_FILE_TOO_LARGE", None),
}
"""Node's code and message for each error code a tool reports.

A `None` message keeps the tool's own, which names the file's size.
"""


@dataclass(frozen=True, kw_only=True)
class FilesystemCapability:
    """Read-only Node `fs` over the agent's files; create it with `fs()`."""

    max_read_bytes: int = _DEFAULT_MAX_READ_BYTES

    def __post_init__(self) -> None:
        """Reject a read limit no file can meet."""
        if self.max_read_bytes < 1:
            msg = f"max_read_bytes must be at least 1, got {self.max_read_bytes}"
            raise ValueError(msg)


def fs(*, max_read_bytes: int = _DEFAULT_MAX_READ_BYTES) -> FilesystemCapability:
    """Give REPL code read-only, Node-style access to the agent's files.

    Code can then call `fs.readFileSync(path, "utf8")`, `fs.promises.readFile`,
    and `readdir`. Results are the files' contents, not
    the formatted text the filesystem tools return to the model.

    Each call goes through the agent's own `read_file` or `ls` tool,
    the same way `tools.*` calls do, so the tools' path checks and deny rules
    apply. Approval gates do not: with `fs()` enabled, code can read paths that
    an `interrupt` rule or `interrupt_on` gates for the model, without a pause,
    even when no `ptc` list is set. `fs` finds the tools in the eval's
    `ToolRuntime`, so a file tool that middleware hides from the model still
    serves `fs`; leave the tool out of the agent to keep code from reading
    through it.

    Tracing callbacks receive each read's raw result, so file contents up to
    `max_read_bytes` appear in traces.

    !!! warning "Beta"

        The `fs` capability is new and its interface may change.

    Args:
        max_read_bytes: Largest file a single read may return, since the whole
            file is copied into the REPL heap and into traces. A larger file
            throws `ERR_FS_FILE_TOO_LARGE`. The backend still loads the whole
            file before the check, so this does not bound host memory.
            Default 8 MiB.

    Returns:
        A capability for `CodeInterpreterMiddleware(capabilities=[...])`.

    Raises:
        ValueError: If `max_read_bytes` is less than 1.
    """
    return FilesystemCapability(max_read_bytes=max_read_bytes)


def request(
    op: str, args: dict[str, Any], tools: Sequence[BaseTool]
) -> tuple[BaseTool, dict[str, Any]] | dict[str, Any]:
    """Return the tool and arguments for one `fs` call, or the error to throw."""
    operation = _OPERATIONS[op]
    if op == "readFile" and args["encoding"] not in _DECODERS:
        msg = f"The argument 'encoding' is invalid. Received {args['encoding']!r}"
        return {"__lc_fs_error": {"code": "ERR_INVALID_ARG_VALUE", "message": msg}}
    tool = next((t for t in tools if t.name == operation.tool), None)
    if tool is None:
        return _error("ENOSYS", f"this agent has no {operation.tool} tool", op, args)
    return tool, operation.arguments(args)


def request_raw_result(
    runtime: ToolRuntime, tool_call_id: str, max_bytes: int
) -> ToolRuntime:
    """Copy `runtime` with a config that asks the file tool for one call's raw result.

    The request names the call, so the tools that call runs do not honor it.
    """
    config = runtime.config or {}
    raw = {"tool_call_id": tool_call_id, "max_bytes": max_bytes}
    configurable = {**config.get("configurable", {}), _RAW_RESULT_CONFIG_KEY: raw}
    return replace(runtime, config={**config, "configurable": configurable})


def response(op: str, args: dict[str, Any], message: Any) -> Any:
    """Turn a file tool's `ToolMessage` into the value the JS shim returns."""
    artifact = getattr(message, "artifact", None)
    if getattr(message, "status", None) == "error":
        reported = artifact["code"] if artifact is not None else None
        code, detail = _NODE_ERRORS.get(reported, ("EIO", None))
        text = str(message.content).removeprefix("Error: ")
        return _error(code, detail or text, op, args)
    if artifact is None:
        detail = (
            f"the {_OPERATIONS[op].tool} tool returned no raw result; "
            "fs needs a deepagents release whose file tools return raw results"
        )
        return _error("ENOSYS", detail, op, args)
    if op == "readFile":
        return _DECODERS[args["encoding"]](artifact["content"])
    return [posixpath.basename(e["path"].rstrip("/")) for e in artifact["entries"]]


def _error(code: str, detail: str, op: str, args: dict[str, Any]) -> dict[str, Any]:
    syscall = _OPERATIONS[op].syscall
    path = args["path"]
    return {
        "__lc_fs_error": {
            "code": code,
            "message": f"{code}: {detail}, {syscall} '{path}'",
            "syscall": syscall,
            "path": path,
        }
    }


SHIM = (
    resources.files("langchain_quickjs")
    .joinpath("_fs_shim.js")
    .read_text(encoding="utf-8")
)
