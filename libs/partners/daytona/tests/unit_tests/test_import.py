from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import langchain_daytona
from langchain_daytona.sandbox import DaytonaSandbox

if TYPE_CHECKING:
    from collections.abc import Callable

ADAPTIVE_POLLING_FAST_THRESHOLD = 1
ADAPTIVE_POLLING_MEDIUM_THRESHOLD = 10
COMMAND_TIMEOUT_EXIT_CODE = 124


def _make_sandbox(
    *,
    sync_polling_interval: float | Callable[[float], float] = 0.1,
) -> tuple[DaytonaSandbox, MagicMock]:
    mock_sdk = MagicMock()
    mock_sdk.id = "sb-123"
    sb = DaytonaSandbox(
        sandbox=mock_sdk,
        sync_polling_interval=sync_polling_interval,
    )
    return sb, mock_sdk


def test_import_daytona() -> None:
    assert langchain_daytona is not None


def test_execute_returns_stdout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.side_effect = [SimpleNamespace(exit_code=0)]
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="hello world", stderr=""
    )

    result = sb.execute("echo hello world")

    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False
    assert mock_sdk.process.create_session.call_count == 1
    assert mock_sdk.process.delete_session.call_count == 1


def test_execute_polls_with_fixed_interval() -> None:
    sb, mock_sdk = _make_sandbox(sync_polling_interval=0.25)
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
    assert [call.args[0] for call in mock_sleep.call_args_list] == [0.25, 0.25]


def test_execute_polls_with_callable_interval() -> None:
    sleep_schedule: list[tuple[float, float]] = []

    def adaptive_interval(elapsed: float) -> float:
        interval = (
            0.1
            if elapsed < ADAPTIVE_POLLING_FAST_THRESHOLD
            else 0.2
            if elapsed < ADAPTIVE_POLLING_MEDIUM_THRESHOLD
            else 1.0
        )
        sleep_schedule.append((elapsed, interval))
        return interval

    sb, mock_sdk = _make_sandbox(sync_polling_interval=adaptive_interval)
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.side_effect = [
        SimpleNamespace(exit_code=None),
        SimpleNamespace(exit_code=None),
        SimpleNamespace(exit_code=None),
        SimpleNamespace(exit_code=0),
    ]
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="done", stderr=""
    )

    with (
        patch("langchain_daytona.sandbox.time.sleep") as mock_sleep,
        patch(
            "langchain_daytona.sandbox.time.monotonic",
            side_effect=[0.0, 0.0, 0.5, 5.0, 65.0],
        ),
    ):
        result = sb.execute("sleep 5", timeout=0)

    assert result.exit_code == 0
    assert sleep_schedule == [(0.0, 0.1), (0.5, 0.1), (5.0, 0.2)]
    assert [call.args[0] for call in mock_sleep.call_args_list] == [0.1, 0.1, 0.2]


def test_execute_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=None)

    with (
        patch("langchain_daytona.sandbox.time.sleep"),
        patch(
            "langchain_daytona.sandbox.time.monotonic",
            side_effect=[0.0, 0.0, 0.0, 11.0],
        ),
    ):
        result = sb.execute("sleep 999", timeout=10)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert "timed out" in result.output
