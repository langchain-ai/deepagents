"""Node-shaped `fs` over the `ptc` file tools: one `_OPERATIONS` row per tool call."""

from __future__ import annotations

import base64
import json
import posixpath
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


def _name(entry: dict[str, Any]) -> str:
    return posixpath.basename(entry["path"].rstrip("/"))


def _check_encoding(args: dict[str, Any], allowed: Collection[str | None]) -> None:
    if args["encoding"] not in allowed:
        msg = f"The argument 'encoding' is invalid. Received {args['encoding']!r}"
        raise ValueError(msg)


def _read(args: dict[str, Any]) -> dict[str, Any]:
    _check_encoding(args, _DECODERS)
    return {"file_path": args["path"]}


def _stat(args: dict[str, Any], raw: Any) -> dict[str, Any]:
    """Find the path among its parent's entries; size and mtime are best-effort."""
    name = posixpath.basename(args["path"])
    for entry in raw["entries"]:
        if _name(entry) == name:
            size, mtime = entry.get("size"), entry.get("modified_at")
            return {"isDir": entry["is_dir"], "size": size, "mtime": mtime}
    raise FileNotFoundError


def _globbed(args: dict[str, Any], raw: Any) -> dict[str, Any]:
    paths = [posixpath.relpath(p, args["path"]) for p in raw["paths"]]
    return {"paths": paths, "truncated": raw["truncated"]}


_OPERATIONS = {
    "readFile": _Operation(
        tool="read_file",
        syscall="open",
        arguments=_read,
        value=lambda a, raw: _DECODERS[a["encoding"]](raw["content"]),
        doc=(
            '- `fs.readFileSync(path, "utf8")` returns the whole file as a string, '
            "with no line numbers or headers, as in "
            '`JSON.parse(fs.readFileSync("/config.json", "utf8"))`. Without an '
            "encoding it returns a `Uint8Array` whose `toString()` decodes UTF-8; "
            '`"base64"` also works.'
        ),
    ),
    "readdir": _Operation(
        tool="ls",
        syscall="scandir",
        arguments=lambda a: {"path": a["path"]},
        value=lambda _, raw: [
            {"name": _name(e), "isDir": e["is_dir"]} for e in raw["entries"]
        ],
        doc=(
            "- `fs.readdirSync(dir)` lists entry names; with `{ withFileTypes: true }` "
            "it returns entries with `name`, `isFile()`, and `isDirectory()`. "
            "`fs.existsSync(path)` checks a path."
        ),
    ),
    "glob": _Operation(
        tool="glob",
        syscall="glob",
        arguments=lambda a: {"pattern": a["pattern"], "path": a["path"]},
        value=_globbed,
        doc=(
            "- `fs.globSync(pattern, { cwd })`, or `for await` over "
            "`fs.promises.glob`, finds files the way your glob tool does."
        ),
    ),
    "stat": _Operation(
        tool="ls",
        syscall="stat",
        arguments=lambda a: {"path": posixpath.dirname(a["path"])},
        value=_stat,
        doc=(
            "- `fs.statSync(path)` gives `isFile()`, `isDirectory()`, and `size` and "
            "`mtime` when known; `fs.accessSync(path)` throws `ENOENT` if it is "
            "missing."
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
    try:
        arguments = operation.arguments(args)
    except ValueError as e:
        return [1, "ERR_INVALID_ARG_VALUE", str(e), {"name": "TypeError"}]
    tool = tools.get(operation.tool)
    if tool is None:
        detail = f"function not implemented; add {operation.tool!r} to `ptc`"
        return _error("ENOSYS", detail, op, args)
    return tool, arguments


def response(op: str, args: dict[str, Any], result: CallResult) -> list[Any]:
    """Turn a file tool's call result into the envelope the JS shim settles."""
    operation = _OPERATIONS[op]
    if result.status == "error":
        return _error(*_failure(result), op, args)
    try:
        return [0, operation.value(args, result.artifact)]
    except FileNotFoundError:
        return _error(*_NODE_ERRORS["file_not_found"], op, args)
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


_MODULE_EXPORTS = {
    "node:fs": (
        "readFileSync",
        "readdirSync",
        "existsSync",
        "globSync",
        "statSync",
        "lstatSync",
        "accessSync",
        "promises",
    ),
    "node:fs/promises": ("readFile", "readdir", "glob", "stat", "lstat", "access"),
    "node:path": (
        "isAbsolute",
        "normalize",
        "join",
        "resolve",
        "dirname",
        "basename",
        "extname",
        "sep",
        "posix",
    ),
}
"""The names each module exports, which an ES module must list statically."""

# QuickJS caches each module per context, while the shim installs a new `fs` every
# eval, so each export looks up the running eval's module when it is called.
_MODULE_SOURCE = """const names = %s;
const values = {
  "node:fs": () => ({ promises: forward("node:fs/promises") }),
  "node:path": (m) => ({ sep: "/", posix: m }),
};
const forward = (id) => {
  const m = Object.fromEntries(names[id].map((key) => [key, (...a) => {
    const live = globalThis.__lc_fs?.[id];
    if (live !== undefined) return live[key](...a);
    const msg = "ENOSYS: function not implemented; list a file tool in `ptc`, ";
    throw Object.assign(new Error(msg + key), { code: "ENOSYS" });
  }]));
  return Object.assign(m, values[id]?.(m));
};
const m = forward(%s);
export default m;
export const { %s } = m;
"""


def normalize_module(base: str, specifier: str) -> str | None:
    """Resolve an `import()` specifier, such as `fs` or `node:fs`, or return `None`."""
    del base
    name = specifier if specifier.startswith("node:") else f"node:{specifier}"
    return name if name in _MODULE_EXPORTS else None


def load_module(name: str) -> str | None:
    """Return the source of the ES module `normalize_module` named."""
    if name not in _MODULE_EXPORTS:
        return None
    exports = ", ".join(_MODULE_EXPORTS[name])
    return _MODULE_SOURCE % (json.dumps(_MODULE_EXPORTS), json.dumps(name), exports)


SHIM = resources.files("langchain_quickjs").joinpath("_fs_shim.js").read_text("utf-8")
"""A JS function expression that installs `fs`; see `_fs_shim.js`."""
