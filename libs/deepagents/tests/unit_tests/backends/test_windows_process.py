"""Windows pipe polling and cross-platform output collection tests."""

import asyncio
import io
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from itertools import chain, repeat
from unittest.mock import MagicMock, patch

import pytest

import deepagents.backends._windows_process as windows_process
from deepagents.backends import local_shell
from deepagents.backends._windows_process import WindowsProcessReader


def _process(*, errors: str = "strict") -> MagicMock:
    return MagicMock(
        args="command",
        stdout=io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors=errors),
        stderr=io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors=errors),
    )


def test_peek_uses_available_byte_count() -> None:
    """Test the byte count is taken from the `PeekNamedPipe` result."""
    # `_peek_pipe` keeps a literal `sys.platform` test so the type checker can
    # drop the Windows-only imports elsewhere, so this test has to patch it.
    # That is safe here only because this test starts no threads and runs no
    # event loop while the patch is active.
    with (
        patch.object(windows_process, "_winapi", create=True) as api,
        patch.object(windows_process, "msvcrt", create=True) as runtime,
        patch.object(windows_process.sys, "platform", "win32"),
    ):
        runtime.get_osfhandle.return_value = 123
        api.PeekNamedPipe.return_value = (7, 0)
        assert windows_process._peek_pipe(42) == 7
        api.PeekNamedPipe.assert_called_once_with(123, 0)


def test_does_not_read_empty_live_pipe() -> None:
    pipe = MagicMock()
    with patch.object(windows_process, "_peek_pipe", return_value=0), patch.object(windows_process.os, "read") as read:
        assert windows_process._read_available(pipe) is None
    read.assert_not_called()


def test_reads_only_available_bytes() -> None:
    pipe = MagicMock()
    with patch.object(windows_process, "_peek_pipe", return_value=3), patch.object(windows_process.os, "read", return_value=b"abc") as read:
        assert windows_process._read_available(pipe) == b"abc"
    read.assert_called_once_with(pipe.fileno(), 3)


@pytest.mark.parametrize("code", [109, 232, 233, 5])
def test_pipe_errors_distinguish_eof_from_failure(code: int) -> None:
    """Test every end-of-stream error code reads as EOF and others propagate.

    Windows reports a pipe whose write end is gone as `ERROR_BROKEN_PIPE` (109),
    `ERROR_NO_DATA` (232) or `ERROR_PIPE_NOT_CONNECTED` (233), depending on how
    far teardown has progressed. Any other code is a real failure.
    """

    class PipeError(OSError):
        winerror = code

    error = PipeError("pipe failure")
    with patch.object(windows_process, "_peek_pipe", side_effect=error):
        if code in windows_process._END_OF_STREAM_ERRORS:
            assert windows_process._read_available(MagicMock()) == b""
        else:
            with pytest.raises(OSError, match="pipe failure"):
                windows_process._read_available(MagicMock())


def test_retries_preserve_multibyte_output_and_newlines() -> None:
    process = _process()
    process.poll.return_value = 7
    reader = WindowsProcessReader(process)
    try:
        with (
            patch.object(windows_process, "_read_available", side_effect=[b"\xc3", b"err\r", b"\xa9\r", b"\n", b"\nx\r", b"", b"\n", b""]),
            patch.object(windows_process.time, "monotonic", side_effect=chain([0, 0], repeat(2))),
        ):
            with pytest.raises(subprocess.TimeoutExpired):
                reader.communicate(timeout=1)
            assert reader.communicate(timeout=1) == ("é\nx\n", "err\n")
        assert process.stdout.closed
        assert process.stderr.closed
    finally:
        process.stdout.close()
        process.stderr.close()


def test_exited_shell_does_not_discard_descendant_output() -> None:
    process = _process()
    process.poll.return_value = 0
    try:
        with (
            patch.object(windows_process, "_read_available", side_effect=[None, None, b"late", b"", b""]),
            patch.object(windows_process.time, "sleep"),
        ):
            assert WindowsProcessReader(process).communicate(timeout=1) == ("late", "")
    finally:
        process.stdout.close()
        process.stderr.close()


def test_closed_pipes_still_wait_for_process_exit() -> None:
    process = _process()
    process.poll.side_effect = [None, 4]
    try:
        with patch.object(windows_process, "_read_available", return_value=b""):
            assert WindowsProcessReader(process).communicate(timeout=1) == ("", "")
        assert process.poll.call_count == 2
    finally:
        process.stdout.close()
        process.stderr.close()


def test_decode_errors_follow_process_settings() -> None:
    process = _process(errors="replace")
    try:
        with patch.object(windows_process, "_read_available", side_effect=[b"\xff", b"", b""]):
            assert WindowsProcessReader(process).communicate(timeout=1) == ("�", "")
    finally:
        process.stdout.close()
        process.stderr.close()


async def _cancel_command(backend: local_shell.LocalShellBackend, started: threading.Event) -> None:
    task = asyncio.create_task(backend.aexecute("command"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert task.cancelled()


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "async-cancellation-and-shutdown"])
def test_backend_shutdown_does_not_wait_for_inherited_pipe_writer(*, cancel: bool) -> None:
    """Test cleanup closes a pipe whose write end another process still holds.

    `subprocess.communicate` would start a reader thread per pipe and that thread
    holds the pipe until the write end closes. `WindowsProcessReader` polls
    instead, so nothing holds the pipe and `_close_pipe` returns at once. `peek`
    reporting zero bytes forever is what an inherited but idle write end looks
    like.
    """
    descriptor, writer = os.pipe()
    pipe = os.fdopen(descriptor, "r", encoding="utf-8")
    process = MagicMock(args="command", pid=1234, stdout=pipe, stderr=None, returncode=0)
    process.poll.return_value = 0
    started = threading.Event()
    polled = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []
    backend = local_shell.LocalShellBackend()

    def peek(_descriptor: int) -> int:
        polled.set()
        started.set()
        return 0

    def run() -> None:
        try:
            if cancel:
                asyncio.run(_cancel_command(backend, started))
            else:
                assert backend.execute("command", timeout=1).exit_code == 124
        except BaseException as error:  # noqa: BLE001  # Report worker assertions and interruptions in the test thread.
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=run, daemon=True)
    with (
        patch.object(local_shell, "_IS_WINDOWS", new=True),
        patch.object(local_shell.subprocess, "Popen", return_value=process),
        patch.object(windows_process, "_peek_pipe", side_effect=peek),
    ):
        worker.start()
        try:
            assert finished.wait(3), "cleanup or executor shutdown waited for the inherited pipe writer"
            assert not errors, errors
            # Without this the test would still pass if the backend silently took
            # the POSIX path and never used the polling reader at all.
            assert polled.is_set(), "the Windows polling reader was never used"
            assert pipe.closed
            # Closing the read end must not disturb the inherited write end.
            os.fstat(writer)
        finally:
            os.close(writer)
            worker.join(5)
            pipe.close()
    assert not worker.is_alive()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows pipe handles")
def test_windows_output_capture() -> None:
    """Test a large two-stream capture survives the polling reader on Windows.

    Both streams exceed the pipe buffer, so the reader has to drain them as the
    process writes. Draining only one would deadlock the other.
    """
    script = "import os; os.write(1, b'hello\\r\\n' * 10000); os.write(2, b'error\\r\\n' * 10000)"
    with subprocess.Popen(  # noqa: S603  # Run a fixed output probe with the test interpreter.
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        try:
            assert WindowsProcessReader(process).communicate(timeout=10) == ("hello\n" * 10000, "error\n" * 10000)
            assert process.returncode == 0
        finally:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows pipe handles")
def test_windows_pipe_close_does_not_wait_for_writer() -> None:
    descriptor, writer = os.pipe()
    pipe = os.fdopen(descriptor, "r", encoding="utf-8")
    process = MagicMock(args="exited shell", stdout=pipe, stderr=None)
    process.poll.return_value = 0
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            WindowsProcessReader(process).communicate(timeout=0.02)
        start = time.monotonic()
        pipe.close()
        assert time.monotonic() - start < 1
        os.fstat(writer)
    finally:
        pipe.close()
        os.close(writer)
