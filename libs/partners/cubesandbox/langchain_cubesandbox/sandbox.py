"""CubeSandbox sandbox backend implementation.

This module wraps the official `cubesandbox` Python SDK into a
[`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol]
backend by subclassing
[`BaseSandbox`][deepagents.backends.sandbox.BaseSandbox]. `BaseSandbox`
provides `ls / read / write / edit / glob / grep` via the `execute()`
primitive, so this class only needs to implement `id`, `execute`,
`upload_files`, and `download_files`.

CubeSandbox's SDK currently exposes `commands.run()` for shell execution and
`run_code()` for Python execution; it does not provide a file-upload API.
We therefore implement upload/download by running short Python helpers via
`run_code()` and shuttling content as base64 inside the script source. This
avoids the shell-escape hazards of the SDK's `commands.run()` (which
ultimately invokes `subprocess.run(..., shell=True)` inside the sandbox).

We also bypass the SDK's `commands.run()` for `execute()`: that helper
appends the integer return code on the same byte stream as the captured
stdout, which corrupts the last line when the command output does not end
with a newline (e.g. ``cat`` on a no-trailing-newline file yielded
``First content0`` instead of ``First content``). Instead, we run our own
Python wrapper via `run_code()` that emits an unambiguous sentinel around
the exit code.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import TYPE_CHECKING, Final

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    import cubesandbox

_TIMEOUT_EXIT_CODE: Final = 124
"""POSIX `timeout(1)` convention for command timeouts."""

_DEFAULT_TIMEOUT_SECONDS: Final = 30 * 60
"""Default per-command timeout (30 minutes), mirroring `langchain-daytona`."""

_DOWNLOAD_FILE_NOT_FOUND: Final = "__CUBESANDBOX_FILE_NOT_FOUND__"
_DOWNLOAD_IS_DIRECTORY: Final = "__CUBESANDBOX_IS_DIRECTORY__"
_DOWNLOAD_PERMISSION_DENIED: Final = "__CUBESANDBOX_PERMISSION_DENIED__"

_HTTPX_TIMEOUT_CLASS_NAMES: Final = frozenset(
    {"ReadTimeout", "WriteTimeout", "PoolTimeout", "ConnectTimeout"},
)
"""`httpx` timeout class names we treat as `execute()` timeouts.

Sniffed by class name to avoid declaring `httpx` as a direct dependency of
this wrapper (the CubeSandbox SDK already pulls it in transitively).
"""

_EXEC_RC_PREFIX: Final = "__CUBE_PARTNER_EXEC_RC_BEGIN__"
_EXEC_RC_SUFFIX: Final = "__CUBE_PARTNER_EXEC_RC_END__"
_EXEC_RC_PATTERN: Final = re.compile(
    rf"{re.escape(_EXEC_RC_PREFIX)}(-?\d+){re.escape(_EXEC_RC_SUFFIX)}",
)
"""Sentinel framing for the exit code emitted by our `execute()` wrapper.

We bypass ``cubesandbox.Sandbox.commands.run`` because that helper appends
``print(_r.returncode)`` immediately after the captured stdout — when the
command's stdout has no trailing newline (e.g. ``cat`` on a file without
``\\n`` at EOF) the exit code digit ends up glued to the last line of
output, then the SDK's host-side splitlines parser fails to separate them
and returns the corrupted text. Using a sentinel makes the boundary
unambiguous regardless of whether the command emitted a trailing newline.
"""


def _build_exec_wrapper(command: str) -> str:
    """Render the in-sandbox Python wrapper used by `execute()`.

    The wrapper runs the user command via ``subprocess.run(..., shell=True)``
    with captured stdout/stderr, emits the captured stdout verbatim, then
    appends ``<PREFIX><rc><SUFFIX>`` to stdout and writes captured stderr to
    its own stream so envd routes it to ``Execution.logs.stderr``.
    """
    return (
        "import subprocess as _sp, sys as _sys\n"
        f"_r = _sp.run({command!r}, shell=True, capture_output=True, text=True)\n"
        "_sys.stdout.write(_r.stdout)\n"
        f"_sys.stdout.write({_EXEC_RC_PREFIX!r})\n"
        "_sys.stdout.write(str(_r.returncode))\n"
        f"_sys.stdout.write({_EXEC_RC_SUFFIX!r})\n"
        "_sys.stderr.write(_r.stderr)\n"
    )


def _extract_exit_code(raw_stdout: str) -> tuple[str, int | None]:
    """Strip the exit-code sentinel from `raw_stdout` and return `(stdout, rc)`.

    If the sentinel is absent (e.g. the wrapper died before reaching the
    final write), returns the input unchanged with `rc=None` so the caller
    can fall back to inspecting `Execution.error`.
    """
    match = _EXEC_RC_PATTERN.search(raw_stdout)
    if match is None:
        return raw_stdout, None
    return raw_stdout[: match.start()] + raw_stdout[match.end() :], int(match.group(1))


def _timeout_response(timeout_seconds: int) -> ExecuteResponse:
    """Build a uniform timeout response for `execute()`."""
    return ExecuteResponse(
        output=f"Command timed out after {timeout_seconds} seconds",
        exit_code=_TIMEOUT_EXIT_CODE,
        truncated=False,
    )


def _format_execute_output(stdout: str | None, stderr: str | None) -> str:
    """Combine stdout and stderr into a single output string.

    Mirrors `langchain-daytona`'s convention so the standard sandbox test
    suite sees a consistent format across backends.
    """
    output = stdout or ""
    if stderr is not None and stderr.strip():
        output += f"\n<stderr>{stderr.strip()}</stderr>"
    return output


def _classify_python_error(name: str | None, value: str | None) -> FileOperationError:
    """Map a CubeSandbox `ExecutionError` to a standardized file error code.

    Args:
        name: Python exception class name (e.g. `FileNotFoundError`).
        value: Exception message (used as a fallback signal).

    Returns:
        A `FileOperationError` literal accepted by the deepagents protocol.
    """
    name = name or ""
    value = (value or "").lower()
    if name == "FileNotFoundError" or "no such file" in value:
        return "file_not_found"
    if name == "PermissionError" or "permission denied" in value:
        return "permission_denied"
    if name == "IsADirectoryError" or "is a directory" in value:
        return "is_directory"
    return "file_not_found"


class CubeSandbox(BaseSandbox):
    """CubeSandbox implementation of `SandboxBackendProtocol`.

    Inherits all file-discovery and content operations from `BaseSandbox`,
    which performs them by delegating to `execute()`. Only the
    command-execution primitive and bulk upload/download are implemented
    natively against the CubeSandbox SDK.

    Example:
        ```python
        import cubesandbox
        from langchain_cubesandbox import CubeSandbox

        sb = cubesandbox.Sandbox.create()
        backend = CubeSandbox(sandbox=sb)
        result = backend.execute("echo hello")
        print(result.output)
        ```
    """

    def __init__(
        self,
        *,
        sandbox: cubesandbox.Sandbox,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Wrap an existing CubeSandbox SDK sandbox.

        Args:
            sandbox: Existing `cubesandbox.Sandbox` instance to wrap. The
                caller retains ownership and is responsible for `kill()`.
            timeout: Default command timeout in seconds used when `execute()`
                is called without an explicit `timeout`.
        """
        self._sandbox = sandbox
        self._default_timeout = timeout

    @property
    def id(self) -> str:
        """Return the underlying CubeSandbox sandbox id."""
        return self._sandbox.sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for completion.

                If `None`, the backend's default timeout is used.

        Returns:
            `ExecuteResponse` with combined output, exit code, and the
                `truncated` flag (always `False` for CubeSandbox today).
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        code = _build_exec_wrapper(command)
        try:
            execution = self._sandbox.run_code(code, timeout=effective_timeout)
        except TimeoutError:
            return _timeout_response(effective_timeout)
        except Exception as exc:
            # CubeSandbox SDK wraps HTTP read timeouts as httpx.ReadTimeout or
            # similar; we cannot import httpx here without adding a dependency,
            # so we sniff the class name. Anything else is re-raised.
            if exc.__class__.__name__ in _HTTPX_TIMEOUT_CLASS_NAMES:
                return _timeout_response(effective_timeout)
            raise

        raw_stdout = "".join(execution.logs.stdout) if execution.logs.stdout else ""
        raw_stderr = "".join(execution.logs.stderr) if execution.logs.stderr else ""
        stdout, exit_code = _extract_exit_code(raw_stdout)
        if exit_code is None:
            # Wrapper never finished (e.g. envd killed it); surface as failure.
            exit_code = 1 if execution.error else 0
        return ExecuteResponse(
            output=_format_execute_output(stdout, raw_stderr),
            exit_code=exit_code,
            truncated=False,
        )

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files into the sandbox via inline base64 + `run_code()`.

        Each file becomes a short Python snippet that base64-decodes the
        content and writes it to disk. This avoids shell quoting issues and
        works for arbitrary binary payloads.

        Args:
            files: Sequence of `(absolute_path, content_bytes)` tuples.

        Returns:
            One `FileUploadResponse` per input, in the same order.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            content_b64 = base64.b64encode(content).decode("ascii")
            code = (
                "import base64, os\n"
                f"_path = {path!r}\n"
                f"_data = base64.b64decode({content_b64!r})\n"
                "_parent = os.path.dirname(_path)\n"
                "if _parent and not os.path.isdir(_parent):\n"
                "    os.makedirs(_parent, exist_ok=True)\n"
                "with open(_path, 'wb') as _f:\n"
                "    _f.write(_data)\n"
            )
            try:
                execution = self._sandbox.run_code(code)
            except Exception as exc:  # noqa: BLE001  # surfaced per-file
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error=f"upload_failed: {exc.__class__.__name__}: {exc}",  # ty: ignore[invalid-argument-type]  # narrative fallback
                    )
                )
                continue
            if execution.error is not None:
                err_key = _classify_python_error(
                    execution.error.name,
                    execution.error.value,
                )
                responses.append(FileUploadResponse(path=path, error=err_key))
            else:
                responses.append(FileUploadResponse(path=path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox via inline base64 + `run_code()`.

        For each path we ask the sandbox to base64-encode the file content and
        emit it on stdout. Errors are signalled by sentinel strings rather
        than raising, so a missing file does not abort the whole batch.

        Args:
            paths: Sequence of absolute paths to read.

        Returns:
            One `FileDownloadResponse` per input, in the same order.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue
            code = (
                "import base64, sys, os\n"
                f"_path = {path!r}\n"
                "try:\n"
                "    if os.path.isdir(_path):\n"
                f"        sys.stdout.write({_DOWNLOAD_IS_DIRECTORY!r})\n"
                "    else:\n"
                "        with open(_path, 'rb') as _f:\n"
                "            _data = _f.read()\n"
                "        sys.stdout.write(base64.b64encode(_data).decode('ascii'))\n"
                "except FileNotFoundError:\n"
                f"    sys.stdout.write({_DOWNLOAD_FILE_NOT_FOUND!r})\n"
                "except PermissionError:\n"
                f"    sys.stdout.write({_DOWNLOAD_PERMISSION_DENIED!r})\n"
            )
            try:
                execution = self._sandbox.run_code(code)
            except Exception as exc:  # noqa: BLE001  # surfaced per-file
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=f"download_failed: {exc.__class__.__name__}: {exc}",  # ty: ignore[invalid-argument-type]  # narrative fallback
                    )
                )
                continue
            if execution.error is not None:
                err_key = _classify_python_error(
                    execution.error.name,
                    execution.error.value,
                )
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=err_key)
                )
                continue
            output = "".join(execution.logs.stdout)
            responses.append(self._decode_download_payload(path, output))
        return responses

    @staticmethod
    def _decode_download_payload(path: str, payload: str) -> FileDownloadResponse:
        """Turn a `run_code` stdout payload into a `FileDownloadResponse`."""
        if payload == _DOWNLOAD_FILE_NOT_FOUND:
            return FileDownloadResponse(path=path, content=None, error="file_not_found")
        if payload == _DOWNLOAD_IS_DIRECTORY:
            return FileDownloadResponse(path=path, content=None, error="is_directory")
        if payload == _DOWNLOAD_PERMISSION_DENIED:
            return FileDownloadResponse(
                path=path, content=None, error="permission_denied"
            )
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return FileDownloadResponse(
                path=path,
                content=None,
                error="file_not_found",
            )
        return FileDownloadResponse(path=path, content=data, error=None)
