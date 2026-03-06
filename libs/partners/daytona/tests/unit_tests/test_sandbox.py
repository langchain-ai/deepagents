"""Tests for DaytonaSandbox session-based execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from langchain_daytona.sandbox import _MAX_POLL_INTERVAL, _POLL_INTERVAL, DaytonaSandbox


def _make_sandbox() -> tuple[DaytonaSandbox, MagicMock]:
    """Create a DaytonaSandbox with a mocked daytona.Sandbox."""
    mock_sdk = MagicMock()
    mock_sdk.id = "sb-123"
    sb = DaytonaSandbox(sandbox=mock_sdk)
    return sb, mock_sdk


def test_execute_creates_session_once() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=0)
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="hello", stderr=""
    )

    sb.execute("echo hello")
    sb.execute("echo world")

    assert mock_sdk.process.create_session.call_count == 1


def test_execute_returns_stdout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=0)
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="hello world", stderr=""
    )

    result = sb.execute("echo hello world")

    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False


def test_execute_returns_combined_stdout_stderr() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=1)
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="partial output", stderr="some error"
    )

    result = sb.execute("bad-command")

    assert result.output == "partial output\nsome error"
    assert result.exit_code == 1


def test_execute_empty_output() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=0)
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="", stderr=""
    )

    result = sb.execute("true")

    assert result.output == ""
    assert result.exit_code == 0


def test_execute_polls_until_exit_code() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.side_effect = [
        SimpleNamespace(exit_code=None),
        SimpleNamespace(exit_code=None),
        SimpleNamespace(exit_code=0),
    ]
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="done", stderr=""
    )

    with patch("langchain_daytona.sandbox.time.sleep") as mock_sleep:
        result = sb.execute("sleep 5")

    assert result.exit_code == 0
    assert result.output == "done"
    assert mock_sdk.process.get_session_command.call_count == 3  # noqa: PLR2004
    assert mock_sleep.call_count == 2  # noqa: PLR2004


def test_execute_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=None)

    with (
        patch("langchain_daytona.sandbox.time.sleep"),
        patch("langchain_daytona.sandbox.time.monotonic", side_effect=[0.0, 0.0, 11.0]),
    ):
        result = sb.execute("sleep 999", timeout=10)

    assert result.exit_code == 124  # noqa: PLR2004
    assert "timed out" in result.output


def test_execute_zero_timeout_waits_indefinitely() -> None:
    """timeout=0 means no deadline; we just poll until done."""
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.side_effect = [
        SimpleNamespace(exit_code=None),
        SimpleNamespace(exit_code=0),
    ]
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="ok", stderr=""
    )

    with patch("langchain_daytona.sandbox.time.sleep"):
        result = sb.execute("long-command", timeout=0)

    assert result.exit_code == 0
    assert result.output == "ok"


def test_execute_uses_run_async() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=0)
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="", stderr=""
    )

    sb.execute("echo hi")

    req = mock_sdk.process.execute_session_command.call_args[0][1]
    assert req.run_async is True


def test_execute_uses_default_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=None)

    with (
        patch("langchain_daytona.sandbox.time.sleep"),
        patch(
            "langchain_daytona.sandbox.time.monotonic",
            side_effect=[0.0, 0.0, 30 * 60 + 1.0],
        ),
    ):
        result = sb.execute("sleep 999")

    assert result.exit_code == 124  # noqa: PLR2004
    assert "1800s" in result.output


def test_poll_interval_backs_off() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")

    poll_results = [SimpleNamespace(exit_code=None)] * 20 + [
        SimpleNamespace(exit_code=0)
    ]
    mock_sdk.process.get_session_command.side_effect = poll_results
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="", stderr=""
    )

    with (
        patch("langchain_daytona.sandbox.time.sleep") as mock_sleep,
        patch("langchain_daytona.sandbox.time.monotonic", return_value=0.0),
    ):
        sb.execute("long-thing", timeout=0)

    sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
    assert sleep_args[0] == pytest.approx(_POLL_INTERVAL)
    for val in sleep_args:
        assert val <= _MAX_POLL_INTERVAL + 0.01


def test_id_property() -> None:
    sb, _ = _make_sandbox()
    assert sb.id == "sb-123"


def test_execute_none_logs() -> None:
    """Logs fields may be None rather than empty string."""
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=0)
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout=None, stderr=None
    )

    result = sb.execute("true")

    assert result.output == ""
    assert result.exit_code == 0
