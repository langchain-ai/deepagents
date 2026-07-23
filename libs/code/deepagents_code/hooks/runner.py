"""Bounded asynchronous command execution for Hooks v2."""

from __future__ import annotations

import asyncio
import contextlib
import json
import ntpath
import os
import signal
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from deepagents_code.hooks.capabilities import DEFAULT_COMMAND_TIMEOUT_SECONDS
from deepagents_code.hooks.env import sanitize_hook_environ
from deepagents_code.hooks.models.adapters import HOOK_WIRE_OUTPUT_ADAPTER
from deepagents_code.hooks.models.domain import HookDiagnostic
from deepagents_code.hooks.models.wire import HookWireOutput

if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from pathlib import Path

    from deepagents_code.hooks.snapshot import HookHandler

MAX_HOOK_OUTPUT_BYTES = 100_000
_READ_CHUNK_BYTES = 8192
_BLOCKING_EXIT_CODE = 2
_TERMINATE_WAIT_TIMEOUT = 2.0


@dataclass(frozen=True, slots=True)
class HandlerResult:
    """Validated output and diagnostics from one command handler."""

    handler_id: str
    output: HookWireOutput | None = None
    diagnostics: tuple[HookDiagnostic, ...] = ()
    plain_output: str | None = None


async def run_command_handler(
    handler: HookHandler,
    payload: bytes,
    *,
    cwd: Path,
    default_timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = MAX_HOOK_OUTPUT_BYTES,
    env: dict[str, str] | None = None,
) -> HandlerResult:
    """Run one hook command with bounded time and captured output.

    Args:
        handler: Snapshotted command handler.
        payload: Validated JSON sent to stdin.
        cwd: Working directory inherited from the invocation.
        default_timeout: Timeout used when the handler has no override.
        max_output_bytes: Maximum retained bytes for each output stream.
        env: Optional environment override. Defaults to a sanitized copy of the
            process environment.

    Returns:
        Validated protocol output and structured diagnostics.

    Raises:
        asyncio.CancelledError: If the caller cancels command execution.
    """
    if handler.argv is not None:
        if not handler.argv or not handler.argv[0].strip():
            return _failure(handler.id, "invalid_command", "Hook argv is empty")
    elif not handler.command.strip():
        return _failure(handler.id, "invalid_command", "Hook command is empty")

    if env is not None:
        launch_env = env
    elif handler.inherit_environ:
        launch_env = dict(os.environ)
    else:
        launch_env = sanitize_hook_environ()

    try:
        # Prefer argv + exec when present so Windows metacharacters in paths are
        # not reinterpreted by cmd.exe. Shell form preserves pipes, redirects,
        # globs, and $VAR expansion for ordinary command hooks.
        if handler.argv is not None:
            process = await asyncio.create_subprocess_exec(
                *handler.argv,
                cwd=cwd,
                env=launch_env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                handler.command,
                cwd=cwd,
                env=launch_env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
    except (OSError, ValueError) as exc:
        return _failure(handler.id, "launch_failed", f"Could not launch hook: {exc}")

    timeout = handler.timeout if handler.timeout is not None else default_timeout
    try:
        stdout, stderr, stdout_truncated, stderr_truncated = await asyncio.wait_for(
            _communicate_bounded(process, payload, max_output_bytes),
            timeout=timeout,
        )
    except TimeoutError:
        await _terminate(process)
        return _failure(
            handler.id,
            "timeout",
            f"Hook exceeded its {timeout:g} second timeout",
        )
    except asyncio.CancelledError:
        await _terminate(process)
        raise
    except (BrokenPipeError, ConnectionResetError) as exc:
        await _terminate(process)
        return _failure(handler.id, "io_failed", f"Hook communication failed: {exc}")

    diagnostics: list[HookDiagnostic] = []
    if stdout_truncated:
        diagnostics.append(
            _diagnostic(
                handler.id,
                "stdout_truncated",
                f"Hook stdout exceeded {max_output_bytes} bytes",
            )
        )
    if stderr_truncated:
        diagnostics.append(
            _diagnostic(
                handler.id,
                "stderr_truncated",
                f"Hook stderr exceeded {max_output_bytes} bytes",
            )
        )

    if process.returncode == _BLOCKING_EXIT_CODE:
        reason = _decode(stderr).strip() or "Hook blocked the operation"
        # Exit 2 ignores stdout JSON. Event-specific interpretation of the
        # synthetic decision:"block" is owned by the capability registry via
        # the reducer.
        return HandlerResult(
            handler_id=handler.id,
            output=HookWireOutput(decision="block", reason=reason),
            diagnostics=tuple(diagnostics),
        )
    if process.returncode != 0:
        diagnostics.append(
            _diagnostic(
                handler.id,
                "nonzero_exit",
                f"Hook exited with status {process.returncode}",
            )
        )
        return HandlerResult(handler_id=handler.id, diagnostics=tuple(diagnostics))
    if not stdout.strip():
        return HandlerResult(handler_id=handler.id, diagnostics=tuple(diagnostics))

    plain = _decode(stdout).strip()
    try:
        decoded = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return HandlerResult(
            handler_id=handler.id,
            plain_output=plain,
            diagnostics=tuple(diagnostics),
        )
    try:
        output = HOOK_WIRE_OUTPUT_ADAPTER.validate_python(decoded)
    except ValidationError as exc:
        diagnostics.append(
            _diagnostic(
                handler.id,
                "invalid_output",
                f"Hook output failed protocol validation: {exc.title}",
            )
        )
        return HandlerResult(handler_id=handler.id, diagnostics=tuple(diagnostics))
    return HandlerResult(
        handler_id=handler.id,
        output=output,
        diagnostics=tuple(diagnostics),
    )


async def _communicate_bounded(
    process: Process,
    payload: bytes,
    limit: int,
) -> tuple[bytes, bytes, bool, bool]:
    if process.stdin is None or process.stdout is None or process.stderr is None:
        msg = "Hook process pipes are unavailable"
        raise BrokenPipeError(msg)
    stdout_task = asyncio.create_task(_read_bounded(process.stdout, limit))
    stderr_task = asyncio.create_task(_read_bounded(process.stderr, limit))
    stdin_task = asyncio.create_task(_write_input(process, payload))
    try:
        await process.wait()
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await stdin_task
    except BaseException:
        stdin_task.cancel()
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(
            stdin_task,
            stdout_task,
            stderr_task,
            return_exceptions=True,
        )
        raise
    return stdout[0], stderr[0], stdout[1], stderr[1]


async def _write_input(process: Process, payload: bytes) -> None:
    if process.stdin is None:
        msg = "Hook process stdin is unavailable"
        raise BrokenPipeError(msg)
    process.stdin.write(payload)
    await process.stdin.drain()
    process.stdin.close()


async def _read_bounded(
    stream: asyncio.StreamReader,
    limit: int,
) -> tuple[bytes, bool]:
    retained = bytearray()
    truncated = False
    while chunk := await stream.read(_READ_CHUNK_BYTES):
        remaining = limit - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    return bytes(retained), truncated


async def _terminate(process: Process) -> None:
    """Kill the hook process tree, then reap the direct child."""
    if os.name == "posix" and process.pid is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            with suppress(OSError):
                process.kill()
    elif os.name == "nt":
        await _terminate_windows_tree(process)
    elif process.returncode is None:
        with suppress(OSError):
            process.kill()
    with suppress(OSError, TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=_TERMINATE_WAIT_TIMEOUT)


async def _terminate_windows_tree(process: Process) -> None:
    if process.pid is None:
        return
    taskkill_path = _windows_taskkill_path()
    taskkill = None
    if taskkill_path is not None:
        with suppress(OSError):
            taskkill = await asyncio.create_subprocess_exec(
                taskkill_path,
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
    if taskkill is not None:
        try:
            returncode = await asyncio.wait_for(
                taskkill.wait(),
                timeout=_TERMINATE_WAIT_TIMEOUT,
            )
        except TimeoutError:
            with suppress(OSError):
                taskkill.kill()
            with suppress(OSError, TimeoutError):
                await asyncio.wait_for(
                    taskkill.wait(),
                    timeout=_TERMINATE_WAIT_TIMEOUT,
                )
        else:
            if returncode == 0:
                return
    if process.returncode is None:
        with suppress(OSError):
            process.kill()


def _windows_taskkill_path() -> str | None:
    system_root = os.environ.get("SYSTEMROOT")
    if not system_root or not ntpath.isabs(system_root):
        return None
    return ntpath.join(ntpath.normpath(system_root), "System32", "taskkill.exe")


def _decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _failure(handler_id: str, code: str, message: str) -> HandlerResult:
    return HandlerResult(
        handler_id=handler_id,
        diagnostics=(_diagnostic(handler_id, code, message),),
    )


def _diagnostic(handler_id: str, code: str, message: str) -> HookDiagnostic:
    return HookDiagnostic(
        code=code,
        severity="warning",
        message=message,
        handler_id=handler_id,
    )
