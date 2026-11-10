from __future__ import annotations

import functools
import posixpath
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import BackendProtocol

HOST_FUNCTION = "__lc_fs"


class FsError(Exception):
    def __init__(self, message: str, *, code: str, **props: str) -> None:
        super().__init__(message)
        self.code = code
        self.props = props

    @property
    def reply(self) -> dict[str, Any]:
        return {"error": {"message": str(self), "code": self.code, **self.props}}


_NODE_CODES = {
    "file_not_found": "ENOENT",
    "is_directory": "EISDIR",
    "permission_denied": "EACCES",
    "invalid_path": "EINVAL",
}

_ERROR_TEXT = {
    "ENOENT": "no such file or directory",
    "EISDIR": "illegal operation on a directory",
    "ENOTDIR": "not a directory",
    "EACCES": "permission denied",
    "EINVAL": "invalid argument",
    "EIO": "i/o error",
    "ENOSYS": "function not implemented",
}


async def _read_file(
    backend: BackendProtocol, op: str, path: str, _args: dict[str, Any]
) -> str:
    if path.endswith("/"):
        raise _system_error(op, path, "ENOTDIR")
    [response] = await backend.adownload_files([path])
    if response.error is not None:
        raise await _failure(backend, op, path, response.error)
    try:
        return (response.content or b"").decode("utf-8")
    except UnicodeDecodeError:
        msg = "The encoded data was not valid for encoding utf-8"
        code = "ERR_ENCODING_INVALID_ENCODED_DATA"
        raise FsError(msg, code=code, name="TypeError") from None


async def _write_file(
    backend: BackendProtocol, op: str, path: str, args: dict[str, Any]
) -> None:
    if path.endswith("/"):
        raise _system_error(op, path, "EISDIR")
    result = await backend.awrite(path, args["data"])
    if result.error is not None:
        raise await _failure(backend, op, path, result.error)


async def _readdir(
    backend: BackendProtocol, op: str, path: str, _args: dict[str, Any]
) -> list[str]:
    if not await _stat(backend, op, path):
        raise _system_error(op, path, "ENOTDIR")
    listing = await backend.als(path)
    if listing.error is not None:
        raise _system_error(op, path, "EIO", listing.error)
    return [posixpath.basename(e["path"].rstrip("/")) for e in listing.entries or ()]


async def _access(
    backend: BackendProtocol, op: str, path: str, _args: dict[str, Any]
) -> None:
    await _stat(backend, op, path)


async def _rm_stat(
    backend: BackendProtocol, op: str, path: str, _args: dict[str, Any]
) -> bool:
    return await _stat(backend, op, path)


async def _rm(
    backend: BackendProtocol, op: str, path: str, _args: dict[str, Any]
) -> None:
    result = await backend.adelete(path)
    if result.error is not None:
        raise _system_error(op, path, "EIO", result.error)


_OPERATIONS: dict[
    str,
    tuple[str, Callable[[BackendProtocol, str, str, dict[str, Any]], Awaitable[Any]]],
] = {
    "readFile": ("open", _read_file),
    "writeFile": ("open", _write_file),
    "readdir": ("scandir", _readdir),
    "access": ("access", _access),
    # The shim checks a path before `rm`, as Node does; Node names `lstat` then.
    "rmStat": ("lstat", _rm_stat),
    "rm": ("rm", _rm),
}


def public_name(op: str) -> str:
    return "fs.rm" if op == "rmStat" else f"fs.{op}"


async def run(backend: BackendProtocol, op: str, args: dict[str, Any]) -> Any:
    """Run one `fs` call against `backend`, raising `FsError` on any failure."""
    if not all(isinstance(args.get(k, ""), str) for k in ("path", "data")):
        msg = 'The "path" and "data" arguments must be of type string'
        raise FsError(msg, code="ERR_INVALID_ARG_TYPE", name="TypeError")
    target = "/" + posixpath.normpath("/" + args["path"]).lstrip("/")
    path = target + "/" if args["path"].endswith("/") and target != "/" else target
    try:
        return await _OPERATIONS[op][1](backend, op, path, args)
    except FsError:
        raise
    except NotImplementedError:
        raise _system_error(
            op, path, "ENOSYS", "the backend does not support it"
        ) from None
    except Exception as e:  # noqa: BLE001  # a failing backend rejects this one fs call
        raise _system_error(op, path, "EIO", str(e)) from None


async def _failure(backend: BackendProtocol, op: str, path: str, error: str) -> FsError:
    code = _NODE_CODES.get(error)
    if code in {"EISDIR", "EACCES"}:
        return _system_error(op, path, code)
    # Backends can report a directory, or a path under a file, as missing or
    # invalid; `_is_directory` raises `ENOTDIR` for the latter.
    if await _is_directory(backend, op, path):
        return _system_error(op, path, "EISDIR")
    return _system_error(op, path, code or "EIO", None if code else error)


async def _stat(backend: BackendProtocol, op: str, path: str) -> bool:
    """Whether `path` is a directory, raising `ENOENT` if nothing is there."""
    is_directory = await _is_directory(backend, op, path)
    if is_directory is None:
        raise _system_error(op, path, "ENOENT")
    if path.endswith("/") and not is_directory:
        raise _system_error(op, path, "ENOTDIR")
    return is_directory


async def _is_directory(backend: BackendProtocol, op: str, path: str) -> bool | None:
    """Whether `path` is a directory, or `None` if nothing is there."""
    target = path.rstrip("/")
    if not target:
        return True
    listing = await backend.als(posixpath.dirname(target))
    if listing.error is not None:
        # A parent that cannot be listed counts as missing.
        raise _system_error(op, path, "ENOENT", listing.error)
    entries = (e for e in listing.entries or () if e["path"].rstrip("/") == target)
    entry = next(entries, None)
    return None if entry is None else entry.get("is_dir", False)


def _system_error(op: str, path: str, code: str, reason: str | None = None) -> FsError:
    syscall = _OPERATIONS[op][0]
    message = f"{code}: {_ERROR_TEXT[code]}, {syscall} '{path}'"
    if reason:
        message += f" ({reason})"
    return FsError(message, code=code, syscall=syscall, path=path)


@functools.cache
def install_source() -> str:
    shim = (
        resources.files("langchain_quickjs").joinpath("_fs_shim.js").read_text("utf-8")
    )
    return f"({shim})({HOST_FUNCTION}); undefined"
