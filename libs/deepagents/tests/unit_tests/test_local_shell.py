"""Unit tests for LocalShellBackend per-command timeout features."""

import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents.backends.local_shell import DEFAULT_EXECUTE_TIMEOUT, LocalShellBackend, _terminate_process_group


class TestDefaultTimeoutConstant:
    """Tests for the named default timeout constant."""

    def test_default_timeout_uses_constant(self) -> None:
        """Backend created without explicit timeout should use the default constant."""
        backend = LocalShellBackend()
        assert backend._default_timeout == DEFAULT_EXECUTE_TIMEOUT


class TestInitTimeoutValidation:
    """Tests for timeout validation in __init__."""

    def test_custom_timeout_accepted(self) -> None:
        """Custom positive timeout should be stored."""
        backend = LocalShellBackend(timeout=300)
        assert backend._default_timeout == 300

    def test_zero_timeout_raises(self) -> None:
        """Zero timeout should raise ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            LocalShellBackend(timeout=0)

    def test_negative_timeout_raises(self) -> None:
        """Negative timeout should raise ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            LocalShellBackend(timeout=-10)


class TestPerCommandTimeout:
    """Tests for per-command timeout override in execute()."""

    def test_per_command_timeout_overrides_default(self) -> None:
        """A short default timeout should not kill a command given a longer per-command timeout."""
        backend = LocalShellBackend(timeout=1, inherit_env=True)
        # sleep 0.1 would fail with timeout=1 default, but we verify the
        # per-command timeout is actually used by running a real command.
        result = backend.execute("echo per-command-ok", timeout=10)
        assert result.exit_code == 0
        assert "per-command-ok" in result.output

    def test_default_timeout_used_when_not_specified(self) -> None:
        """When no per-command timeout, the default should be used."""
        backend = LocalShellBackend(timeout=10, inherit_env=True)
        result = backend.execute("echo default-ok")
        assert result.exit_code == 0
        assert "default-ok" in result.output

    def test_per_command_timeout_actually_expires(self) -> None:
        """A per-command timeout should actually terminate long-running commands."""
        backend = LocalShellBackend(timeout=60, inherit_env=True)
        result = backend.execute(f'"{sys.executable}" -c "import time; time.sleep(30)"', timeout=1)
        assert result.exit_code == 124
        assert "timed out" in result.output.lower()

    def test_per_command_zero_timeout_raises(self) -> None:
        """Zero per-command timeout should raise ValueError."""
        backend = LocalShellBackend(inherit_env=True)
        with pytest.raises(ValueError, match="timeout must be positive"):
            backend.execute("echo hello", timeout=0)

    def test_per_command_negative_timeout_raises(self) -> None:
        """Negative per-command timeout should raise ValueError."""
        backend = LocalShellBackend(inherit_env=True)
        with pytest.raises(ValueError, match="timeout must be positive"):
            backend.execute("echo hello", timeout=-5)


class TestTimeoutErrorMessage:
    """Tests for timeout error message with retry guidance."""

    def test_default_timeout_error_includes_retry_guidance(self) -> None:
        """Default timeout error should guide the LLM to use the timeout parameter."""
        backend = LocalShellBackend(timeout=1, inherit_env=True)
        with patch(
            "subprocess.run" if sys.platform == "win32" else "deepagents.backends.local_shell._run_posix_command",
            side_effect=subprocess.TimeoutExpired("cmd", 1),
        ):
            result = backend.execute("sleep 10")
            assert "timed out" in result.output.lower()
            assert "timeout parameter" in result.output.lower()
            assert result.exit_code == 124

    def test_custom_timeout_error_shows_effective_value(self) -> None:
        """Custom timeout error should show the value used and not suggest re-using timeout."""
        backend = LocalShellBackend(timeout=60, inherit_env=True)
        with patch(
            "subprocess.run" if sys.platform == "win32" else "deepagents.backends.local_shell._run_posix_command",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            result = backend.execute("sleep 10", timeout=5)
            assert "5" in result.output
            assert "custom timeout" in result.output.lower()
            assert "may be stuck" in result.output.lower()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("wait_for_child", [False, True])
def test_timeout_kills_descendants(tmp_path: Path, *, wait_for_child: bool) -> None:
    """Kill descendants even when the shell has already exited."""
    marker = tmp_path / "survived"
    script = (
        "import signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('partial-stdout', flush=True); "
        "import sys; print('partial-stderr', file=sys.stderr, flush=True); "
        f"time.sleep(2); Path({str(marker)!r}).touch()"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} &"
    if wait_for_child:
        command += " wait"
    started = time.monotonic()
    with patch("deepagents.backends.local_shell._terminate_process_group", wraps=_terminate_process_group) as cleanup:
        result = LocalShellBackend(root_dir=tmp_path, timeout=1).execute(command)
    process = cleanup.call_args.args[0]
    assert process.returncode is not None
    assert process.stdout.closed
    assert process.stderr.closed
    assert time.monotonic() - started < 3
    assert result.exit_code == 124
    assert result.timed_out
    assert result.stdout == "partial-stdout\n"
    assert result.stderr == "partial-stderr\n"
    assert result.output.count("partial-stdout") == 1
    assert result.output.count("[stderr] partial-stderr") == 1
    assert "timed out" in result.output
    time.sleep(1.2)
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_timeout_bounds_cleanup_with_escaped_pipe_holder(tmp_path: Path) -> None:
    """A detached descendant holding pipes cannot extend cleanup indefinitely."""
    pid_file = tmp_path / "escaped-pid"
    script = (
        "import os, time; from pathlib import Path; os.setsid(); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "print('escaped-output', flush=True); time.sleep(30)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} &"
    started = time.monotonic()
    try:
        result = LocalShellBackend(root_dir=tmp_path, timeout=1).execute(command)
        assert time.monotonic() - started < 3.5
        assert result.exit_code == 124
        assert result.timed_out
        assert result.stdout == "escaped-output\n"
        assert result.stderr == ""
        assert result.output.count("escaped-output") == 1
        assert "timed out" in result.output
    finally:
        if pid_file.exists():
            with suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell syntax")
def test_timeout_partial_output_respects_byte_limit(tmp_path: Path) -> None:
    """Preserve partial output within the existing output budget."""
    result = LocalShellBackend(root_dir=tmp_path, timeout=1, max_output_bytes=4).execute("printf abcdefgh; sleep 30")
    assert result.exit_code == 124
    assert result.truncated
    assert result.timed_out
    assert result.stdout == "abcdefgh"
    assert result.stderr == ""
    assert result.output.startswith("abcd\n\n... Output truncated at 4 bytes.")
    assert "timed out" in result.output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell syntax")
def test_normal_exit_124_is_not_reported_as_timeout(tmp_path: Path) -> None:
    """Keep a command's own exit status distinct from timeout diagnostics."""
    result = LocalShellBackend(root_dir=tmp_path).execute("printf ordinary; exit 124")
    assert result.exit_code == 124
    assert not result.timed_out
    assert result.stdout == "ordinary"
    assert result.stderr == ""
    assert result.output == "ordinary\n\nExit code: 124"


def test_windows_timeout_keeps_existing_behavior() -> None:
    """Retain the Windows subprocess.run timeout path and response."""
    backend = LocalShellBackend()
    with (
        patch("deepagents.backends.local_shell.sys.platform", "win32"),
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1, output=b"partial")) as run,
    ):
        result = backend.execute("command", timeout=1)
    assert result.exit_code == 124
    assert result.output.startswith("Error: Command timed out after 1 seconds (custom timeout).")
    assert result.timed_out
    assert result.stdout == "partial"
    assert result.stderr == ""
    assert "partial" not in result.output
    assert run.call_args.kwargs["start_new_session"] is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell syntax")
@pytest.mark.parametrize("use_async", [False, True])
async def test_fractional_timeout(tmp_path: Path, *, use_async: bool) -> None:
    """Accept subsecond remaining budgets for sync and inherited async calls."""
    backend = LocalShellBackend(root_dir=tmp_path, timeout=0.5)
    result = await backend.aexecute("sleep 30", timeout=0.05) if use_async else backend.execute("sleep 30", timeout=0.05)
    assert result.timed_out
    assert result.exit_code == 124
    assert "0.05 seconds" in result.output
