"""Windows pipe polling and cross-platform output collection tests."""

import asyncio
import io
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
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


def test_caps_each_read() -> None:
    pipe = MagicMock()
    with patch.object(windows_process, "_peek_pipe", return_value=1_000_000), patch.object(windows_process.os, "read", return_value=b"x") as read:
        windows_process._read_available(pipe)
    assert read.call_args.args[1] == 32_768


@pytest.mark.parametrize("code", [109, 5])
def test_pipe_errors_distinguish_eof_from_failure(code: int) -> None:
    class PipeError(OSError):
        winerror = code

    error = PipeError("pipe failure")
    with patch.object(windows_process, "_peek_pipe", side_effect=error):
        if code == 109:
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
            patch.object(windows_process.time, "monotonic", side_effect=[0, 0, 2, 2, 2, 2, 2, 2]),
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


def test_exited_shell_with_live_empty_pipes_times_out() -> None:
    process = _process()
    process.poll.return_value = 0
    try:
        with patch.object(windows_process, "_read_available", return_value=None):
            start = time.monotonic()
            with pytest.raises(subprocess.TimeoutExpired):
                WindowsProcessReader(process).communicate(timeout=0.02)
            assert time.monotonic() - start < 1
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
    descriptor, writer = os.pipe()
    pipe = os.fdopen(descriptor, "r", encoding="utf-8")
    process = MagicMock(args="command", pid=1234, stdout=pipe, stderr=None, returncode=0)
    process.poll.return_value = 0
    started = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []
    reader = threading.Thread(target=pipe.read, daemon=True)
    backend = local_shell.LocalShellBackend()

    def threaded_communicate(*, timeout: float) -> tuple[str, str]:
        if reader.ident is None:
            reader.start()
        started.set()
        reader.join(timeout)
        if reader.is_alive():
            raise subprocess.TimeoutExpired(process.args, timeout)
        return "", ""

    def peek(_descriptor: int) -> int:
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
    process.communicate.side_effect = threaded_communicate
    with (
        patch.object(local_shell.sys, "platform", "win32"),
        patch.object(local_shell.subprocess, "Popen", return_value=process),
        patch.object(windows_process, "_peek_pipe", side_effect=peek),
    ):
        worker.start()
        try:
            assert finished.wait(3), "cleanup or executor shutdown waited for the inherited pipe writer"
            assert not errors, errors
            assert pipe.closed
            os.fstat(writer)
        finally:
            os.close(writer)
            worker.join(5)
            if reader.ident is not None:
                reader.join(5)
            pipe.close()
    assert not worker.is_alive()
    assert not reader.is_alive()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows pipe handles")
def test_windows_output_capture() -> None:
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


def _wait_for_file(path: Path) -> bool:
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    return path.exists()


async def _cancel_native_command(backend: local_shell.LocalShellBackend, command: str, ready: threading.Event) -> None:
    task = asyncio.create_task(backend.aexecute(command))
    try:
        assert await asyncio.to_thread(ready.wait, 5)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert task.cancelled()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows subprocess inheritance")
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "async-cancellation-and-shutdown"])
def test_windows_backend_shutdown_with_live_descendant(tmp_path: Path, *, cancel: bool) -> None:
    ready, release, done = (tmp_path / name for name in ("ready", "release", "done"))
    child = (
        "from pathlib import Path; import time\n"
        "Path('ready').touch()\n"
        "deadline = time.monotonic() + 15\n"
        "while not Path('release').exists() and time.monotonic() < deadline:\n"
        "    time.sleep(.01)\n"
        "Path('done').touch()\n"
    )
    parent = f"import os, subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}], stdout=sys.stdout, stderr=sys.stderr); os._exit(0)"
    command = subprocess.list2cmdline([sys.executable, "-c", parent])
    backend = local_shell.LocalShellBackend(root_dir=tmp_path, inherit_env=True)
    created = threading.Event()
    cancel_ready = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            if cancel:
                asyncio.run(_cancel_native_command(backend, command, cancel_ready))
            else:
                assert backend.execute(command, timeout=1).exit_code == 124
        except BaseException as error:  # noqa: BLE001  # Forward worker failures to the test thread after releasing the descendant.
            errors.append(error)
        finally:
            finished.set()

    processes: list[subprocess.Popen[str]] = []
    popen = subprocess.Popen

    def capture_process(*args: object, **kwargs: object) -> subprocess.Popen[str]:
        process = popen(*args, **kwargs)
        processes.append(process)
        created.set()
        return process

    worker = threading.Thread(target=run, daemon=True)
    with patch.object(local_shell.subprocess, "Popen", side_effect=capture_process):
        worker.start()
        try:
            assert created.wait(5), "backend did not create the shell"
            assert _wait_for_file(ready), "descendant did not acquire the inherited output handles"
            assert processes[0].wait(timeout=3) == 0, "direct shell did not exit before descendant cleanup"
            cancel_ready.set()
            assert finished.wait(5), "cleanup or executor shutdown waited for the live descendant"
            assert not errors, errors
            assert not done.exists(), "descendant released its handles before shutdown completed"
        finally:
            release.touch()
            cancel_ready.set()
            worker.join(10)
            assert _wait_for_file(done), "descendant did not exit after release"
    assert not worker.is_alive()


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
