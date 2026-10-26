from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import daytona
import httpx
from daytona._sync.process import Process as SyncProcess
from daytona.common.errors import (
    DaytonaConnectionTimeoutError,
)
from daytona_toolbox_api_client import ApiClient, Configuration, ProcessApi
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.exceptions import ReadTimeoutError

import langchain_daytona
from langchain_daytona.sandbox import DaytonaSandbox

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

ADAPTIVE_POLLING_FAST_THRESHOLD = 1
ADAPTIVE_POLLING_MEDIUM_THRESHOLD = 10
COMMAND_TIMEOUT_EXIT_CODE = 124
COMMAND_TIMEOUT_SECONDS = 2
FINAL_LOG_TIMEOUT_SECONDS = 10
EXPECTED_LOG_RETRIEVAL_ATTEMPTS = 2
CLEANUP_GRACE_SECONDS = 2
_READ_TIMEOUT_MESSAGE = "Read timed out"


def _timeout_seconds(value: object) -> float | None:
    """Extract a numeric timeout from an urllib3 timeout value."""
    if isinstance(value, (int, float)):
        return float(value)

    total = getattr(value, "total", None)
    if isinstance(total, (int, float)):
        return float(total)

    return None


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


def _make_sdk_process(
    process_api: ProcessApi,
    http_client: httpx.Client,
) -> SyncProcess:
    """Create a real Daytona Process across supported SDK versions."""
    parameters = inspect.signature(SyncProcess).parameters
    process_factory = cast("Callable[..., SyncProcess]", SyncProcess)

    if "language" in parameters:
        return process_factory("python", process_api, http_client)

    return process_factory(process_api, http_client)


class _FakeClock:
    """Deterministic stand-in for `time.monotonic` and `time.sleep`."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@contextmanager
def _patched_clock(clock: _FakeClock) -> Iterator[None]:
    with (
        patch("langchain_daytona.sandbox.time.monotonic", clock.monotonic),
        patch("langchain_daytona.sandbox.time.sleep", clock.sleep),
    ):
        yield


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

    monotonic_values = iter((0.0, 0.0, 0.5, 5.0, 65.0))

    def monotonic_side_effect() -> float:
        return next(monotonic_values, 65.0)

    with (
        patch("langchain_daytona.sandbox.time.sleep") as mock_sleep,
        patch(
            "langchain_daytona.sandbox.time.monotonic",
            side_effect=monotonic_side_effect,
        ),
    ):
        result = sb.execute("sleep 5", timeout=0)

    assert result.exit_code == 0
    assert sleep_schedule == [(0.0, 0.1), (0.5, 0.1), (5.0, 0.2)]
    assert [call.args[0] for call in mock_sleep.call_args_list] == [0.1, 0.1, 0.2]


def test_sdk_startup_timeout_uses_remaining_budget() -> None:
    """The real Daytona SDK must convert a stalled command-start request."""
    generated_client = ApiClient(Configuration(host="https://example.invalid"))
    process_api = ProcessApi(generated_client)
    http_client = httpx.Client()
    process = _make_sdk_process(process_api, http_client)
    sandbox = SimpleNamespace(id="test-sandbox", process=process)
    backend = DaytonaSandbox(sandbox=cast("daytona.Sandbox", sandbox))
    observed_timeouts: list[object] = []

    def stalled_transport(*_args: object, **kwargs: object) -> None:
        observed_timeouts.append(_timeout_seconds(kwargs["timeout"]))
        raise ReadTimeoutError(
            HTTPConnectionPool("example.invalid"),
            "/sessions/commands",
            "simulated read timeout",
        )

    try:
        with (
            patch(
                "langchain_daytona.sandbox.time.monotonic",
                return_value=100.0,
            ),
            patch.object(process, "create_session"),
            patch.object(process, "delete_session"),
            patch.object(
                generated_client.rest_client.pool_manager,
                "request",
                side_effect=stalled_transport,
            ),
        ):
            result = backend.execute("echo started", timeout=2)
    finally:
        http_client.close()

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert result.output == "Command timed out after 2 seconds"
    assert observed_timeouts == [2.0]


def test_execute_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    clock = _FakeClock()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=None)
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="", stderr=""
    )

    with _patched_clock(clock):
        result = sb.execute("sleep 999", timeout=10)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert result.output == "Command timed out after 10 seconds"
    assert mock_sdk.process.delete_session.call_count == 1


def test_execute_timeout_keeps_output_produced_before_deadline() -> None:
    sb, mock_sdk = _make_sandbox()
    clock = _FakeClock()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=None)
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="started\n", stderr="warning"
    )

    with _patched_clock(clock):
        result = sb.execute("echo started; sleep 30", timeout=3)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert result.output.startswith("Command timed out after 3 seconds")
    assert "started" in result.output
    assert "<stderr>warning</stderr>" in result.output
    # Logs must be read before the session that holds them is deleted.
    calls = [call[0] for call in mock_sdk.process.method_calls]
    assert calls.index("get_session_command_logs") < calls.index("delete_session")


def test_execute_timeout_still_reported_when_partial_output_unreadable() -> None:
    sb, mock_sdk = _make_sandbox()
    clock = _FakeClock()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=None)
    mock_sdk.process.get_session_command_logs.side_effect = daytona.DaytonaError(
        "logs unavailable"
    )

    with _patched_clock(clock):
        result = sb.execute("sleep 999", timeout=10)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert result.output == "Command timed out after 10 seconds"
    assert mock_sdk.process.delete_session.call_count == 1


def test_execute_bounds_every_request_by_the_time_left() -> None:
    sb, mock_sdk = _make_sandbox()
    clock = _FakeClock()
    timeout = 10
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.side_effect = [
        SimpleNamespace(exit_code=None),
        SimpleNamespace(exit_code=0),
    ]
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="ok", stderr=""
    )

    with _patched_clock(clock):
        result = sb.execute("true", timeout=timeout)

    process = mock_sdk.process
    assert result.exit_code == 0
    request_timeouts = [
        process.create_session.call_args.kwargs["request_timeout"],
        *[
            c.kwargs["request_timeout"]
            for c in process.get_session_command.call_args_list
        ],
        process.get_session_command_logs.call_args.kwargs["request_timeout"],
        process.delete_session.call_args.kwargs["request_timeout"],
    ]
    assert all(t is not None and t > 0 for t in request_timeouts)
    assert all(t <= timeout for t in request_timeouts)
    # The poll that happens later is given less time than the first one.
    first, second = (
        c.kwargs["request_timeout"] for c in process.get_session_command.call_args_list
    )
    assert second < first


def test_execute_without_deadline_sends_unbounded_command_requests() -> None:
    """`timeout=0` means "wait indefinitely", so no client-side limit is imposed."""
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.side_effect = [SimpleNamespace(exit_code=0)]
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="ok", stderr=""
    )

    result = sb.execute("true", timeout=0)

    process = mock_sdk.process
    assert result.exit_code == 0
    assert process.create_session.call_args.kwargs["request_timeout"] is None
    assert process.get_session_command.call_args.kwargs["request_timeout"] is None
    assert process.get_session_command_logs.call_args.kwargs["request_timeout"] is None
    assert process.execute_session_command.call_args.kwargs["timeout"] == 0


def test_execute_result_survives_session_cleanup_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.side_effect = [SimpleNamespace(exit_code=0)]
    mock_sdk.process.get_session_command_logs.return_value = SimpleNamespace(
        stdout="done", stderr=""
    )
    mock_sdk.process.delete_session.side_effect = daytona.DaytonaTimeoutError(
        "Read timed out"
    )

    result = sb.execute("true")

    assert result.output == "done"
    assert result.exit_code == 0


def test_execute_timeout_during_final_logs_recovers_partial_output() -> None:
    """A final-log timeout returns exit 124 and tries log recovery within grace."""
    sb, mock_sdk = _make_sandbox()
    clock = _FakeClock()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")
    mock_sdk.process.get_session_command.return_value = SimpleNamespace(exit_code=0)

    def logs_then_stall(
        _session_id: str, _cmd_id: str, *, request_timeout: float
    ) -> SimpleNamespace:
        if mock_sdk.process.get_session_command_logs.call_count == 1:
            clock.now += request_timeout
            raise DaytonaConnectionTimeoutError(_READ_TIMEOUT_MESSAGE)
        return SimpleNamespace(stdout="partial output", stderr="")

    mock_sdk.process.get_session_command_logs.side_effect = logs_then_stall

    with _patched_clock(clock):
        result = sb.execute("echo partial output", timeout=2)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert result.output.startswith("Command timed out after 2 seconds")
    assert "partial output" in result.output
    logs_mock = mock_sdk.process.get_session_command_logs
    assert logs_mock.call_count == EXPECTED_LOG_RETRIEVAL_ATTEMPTS
    first, cleanup = logs_mock.call_args_list
    assert first.kwargs["request_timeout"] == COMMAND_TIMEOUT_SECONDS
    assert cleanup.kwargs["request_timeout"] == 1.0


def test_sdk_final_log_timeout_is_converted() -> None:
    """A transport timeout in the real SDK log method returns exit 124."""
    generated_client = ApiClient(Configuration(host="https://example.invalid"))
    process_api = ProcessApi(generated_client)
    http_client = httpx.Client()
    process = _make_sdk_process(process_api, http_client)
    sandbox = SimpleNamespace(id="test-sandbox", process=process)
    backend = DaytonaSandbox(sandbox=cast("daytona.Sandbox", sandbox))
    observed_timeouts: list[float | None] = []

    def stalled_transport(*_args: object, **kwargs: object) -> None:
        observed_timeouts.append(_timeout_seconds(kwargs["timeout"]))
        raise ReadTimeoutError(
            HTTPConnectionPool("example.invalid"),
            "/sessions/command-logs",
            "simulated read timeout",
        )

    try:
        with (
            patch(
                "langchain_daytona.sandbox.time.monotonic",
                return_value=100.0,
            ),
            patch.object(process, "create_session"),
            patch.object(
                process,
                "execute_session_command",
                return_value=SimpleNamespace(cmd_id="c1"),
            ),
            patch.object(
                process,
                "get_session_command",
                return_value=SimpleNamespace(exit_code=0),
            ),
            patch.object(process, "delete_session"),
            patch.object(
                generated_client.rest_client.pool_manager,
                "request",
                side_effect=stalled_transport,
            ),
        ):
            result = backend.execute("echo started", timeout=10)
    finally:
        http_client.close()

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert result.output == "Command timed out after 10 seconds"
    assert observed_timeouts == [10.0, 1.0]


def test_execute_timeout_cleanup_shares_one_grace_budget() -> None:
    """Stalled logs and delete share one two-second cleanup budget."""
    sb, mock_sdk = _make_sandbox()
    clock = _FakeClock()
    mock_sdk.process.execute_session_command.return_value = SimpleNamespace(cmd_id="c1")

    def stalled_status(
        _session_id: str, _cmd_id: str, *, request_timeout: float
    ) -> None:
        clock.now += request_timeout
        raise DaytonaConnectionTimeoutError(_READ_TIMEOUT_MESSAGE)

    def stalled_logs(_session_id: str, _cmd_id: str, *, request_timeout: float) -> None:
        clock.now += request_timeout
        raise DaytonaConnectionTimeoutError(_READ_TIMEOUT_MESSAGE)

    def stalled_delete(_session_id: str, *, request_timeout: float) -> None:
        clock.now += request_timeout
        raise DaytonaConnectionTimeoutError(_READ_TIMEOUT_MESSAGE)

    mock_sdk.process.get_session_command.side_effect = stalled_status
    mock_sdk.process.get_session_command_logs.side_effect = stalled_logs
    mock_sdk.process.delete_session.side_effect = stalled_delete

    with _patched_clock(clock):
        result = sb.execute("sleep 30", timeout=2)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert clock.now == COMMAND_TIMEOUT_SECONDS + CLEANUP_GRACE_SECONDS
    assert (
        mock_sdk.process.get_session_command_logs.call_args.kwargs["request_timeout"]
        == 1.0
    )
    assert mock_sdk.process.delete_session.call_args.kwargs["request_timeout"] == 1.0
