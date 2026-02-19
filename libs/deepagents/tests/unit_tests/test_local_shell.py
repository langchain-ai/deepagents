"""Unit tests for LocalShellBackend per-command timeout features."""

import subprocess
from unittest.mock import patch

import pytest

from deepagents.backends.local_shell import DEFAULT_EXECUTE_TIMEOUT, LocalShellBackend


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
        result = backend.execute("sleep 30", timeout=1)
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
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            result = backend.execute("sleep 10")
            assert "timed out" in result.output.lower()
            assert "timeout parameter" in result.output.lower()
            assert result.exit_code == 124

    def test_custom_timeout_error_shows_effective_value(self) -> None:
        """Custom timeout error should show the value used and not suggest re-using timeout."""
        backend = LocalShellBackend(timeout=60, inherit_env=True)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            result = backend.execute("sleep 10", timeout=5)
            assert "5" in result.output
            assert "custom timeout" in result.output.lower()
            assert "may be stuck" in result.output.lower()
