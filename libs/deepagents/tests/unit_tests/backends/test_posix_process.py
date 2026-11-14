"""POSIX output capture and cancellation-polling regression tests."""

import os
import shlex
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from deepagents.backends._posix_process import PosixProcessReader
from deepagents.backends.local_shell import LocalShellBackend

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX pipe selection")


def test_retries_preserve_multibyte_output_and_newlines() -> None:
    stdout, stdout_writer = os.pipe()
    stderr, stderr_writer = os.pipe()
    process = MagicMock(
        args="command",
        stdout=os.fdopen(stdout, "r", encoding="utf-8"),
        stderr=os.fdopen(stderr, "r", encoding="utf-8", errors="replace"),
    )
    reader = PosixProcessReader(process)
    try:
        try:
            for out, err in [(b"\xc3", b"err\r"), (b"\xa9\r", b"\n\xff"), (b"\nx\r\n", b"\r")]:
                os.write(stdout_writer, out)
                os.write(stderr_writer, err)
                with pytest.raises(subprocess.TimeoutExpired) as caught:
                    reader.communicate(timeout=0.01)
                assert caught.value.output is None
                assert caught.value.stderr is None
        finally:
            os.close(stdout_writer)
            os.close(stderr_writer)
        assert reader.communicate(timeout=1) == ("é\nx\n", "err\n�\n")
        assert process.stdout.closed
        assert process.stderr.closed
    finally:
        process.stdout.close()
        process.stderr.close()


async def test_verbose_command_does_not_copy_output_into_polling_timeouts() -> None:
    """Count discarded snapshot bytes, avoiding machine-dependent timing limits."""
    snapshot_sizes: list[int] = []
    initialize_timeout = subprocess.TimeoutExpired.__init__

    def record_timeout(error, cmd, timeout, output=None, stderr=None) -> None:
        snapshot_sizes.append(len(output or b"") + len(stderr or b""))
        initialize_timeout(error, cmd, timeout, output=output, stderr=stderr)

    size = 1024 * 1024
    script = f"import os, time; os.write(1, b'o' * {size}); os.write(2, b'e' * {size}); time.sleep(0.35)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    with patch.object(subprocess.TimeoutExpired, "__init__", record_timeout):
        result = await LocalShellBackend(max_output_bytes=3 * size).aexecute(command, timeout=5)

    assert result.exit_code == 0
    assert result.output == "o" * size + "\n[stderr] " + "e" * size
    assert not result.truncated
    assert len(snapshot_sizes) >= 2, "the command must span multiple cancellation polls"
    assert sum(snapshot_sizes) == 0, "polling copied accumulated output into discarded timeout exceptions"


def test_closed_pipes_still_wait_for_process_exit() -> None:
    script = "import os, time; os.close(1); os.close(2); time.sleep(60)"
    with subprocess.Popen(  # noqa: S603  # Fixed probe using the test interpreter.
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        reader = PosixProcessReader(process)
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                reader.communicate(timeout=0.5)
            assert process.stdout is not None and process.stdout.closed
            assert process.stderr is not None and process.stderr.closed
            assert process.poll() is None
        finally:
            process.kill()
            process.wait(timeout=5)
        assert reader.communicate(timeout=1) == ("", "")
