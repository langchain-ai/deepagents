"""Unit tests for CLIShellBackend per-command timeout features."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from deepagents_cli.backends import DEFAULT_EXECUTE_TIMEOUT, CLIShellBackend


class TestDefaultTimeoutConstant:
    """Tests for the named default timeout constant."""

    def test_default_timeout_uses_constant(self) -> None:
        """Backend created without explicit timeout should use the default constant."""
        backend = CLIShellBackend()
        assert backend._timeout == DEFAULT_EXECUTE_TIMEOUT


class TestPerCommandTimeout:
    """Tests for per-command timeout override in execute()."""

    def test_per_command_timeout_used(self) -> None:
        """When timeout is passed to execute(), it should override the default."""
        backend = CLIShellBackend(timeout=10, inherit_env=True)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="echo hello",
                returncode=0,
                stdout="hello\n",
                stderr="",
            )
            backend.execute("echo hello", timeout=300)
            _, kwargs = mock_run.call_args
            assert kwargs["timeout"] == 300

    def test_default_timeout_when_not_specified(self) -> None:
        """When no per-command timeout, the default should be used."""
        backend = CLIShellBackend(timeout=60, inherit_env=True)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="echo hello",
                returncode=0,
                stdout="hello\n",
                stderr="",
            )
            backend.execute("echo hello")
            _, kwargs = mock_run.call_args
            assert kwargs["timeout"] == 60

    def test_per_command_zero_timeout_raises(self) -> None:
        """Zero per-command timeout should raise ValueError."""
        backend = CLIShellBackend(inherit_env=True)
        with pytest.raises(ValueError, match="timeout must be positive"):
            backend.execute("echo hello", timeout=0)

    def test_per_command_negative_timeout_raises(self) -> None:
        """Negative per-command timeout should raise ValueError."""
        backend = CLIShellBackend(inherit_env=True)
        with pytest.raises(ValueError, match="timeout must be positive"):
            backend.execute("echo hello", timeout=-5)


class TestTimeoutErrorMessage:
    """Tests for timeout error message with retry guidance."""

    def test_timeout_error_includes_retry_guidance(self) -> None:
        """Timeout error message should include guidance to use timeout parameter."""
        backend = CLIShellBackend(timeout=1, inherit_env=True)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            result = backend.execute("sleep 10")
            assert "timed out" in result.output.lower()
            assert "timeout parameter" in result.output.lower()
            assert result.exit_code == 124

    def test_timeout_error_shows_effective_timeout(self) -> None:
        """Timeout error should show the effective timeout value used."""
        backend = CLIShellBackend(timeout=60, inherit_env=True)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            result = backend.execute("sleep 10", timeout=5)
            assert "5" in result.output
            assert "timeout parameter" in result.output.lower()
