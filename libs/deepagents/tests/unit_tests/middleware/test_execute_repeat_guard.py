"""Tests for the `execute` tool anti-repeat / failure-adaptation guard.

When the same normalized command is issued `_EXECUTE_REPEAT_LIMIT` times in a
row, the wrapper short-circuits and returns adaptation guidance instead of
re-running the command on the backend.
"""

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage

from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.backends.state import StateBackend
from deepagents.middleware.filesystem import (
    _EXECUTE_REPEAT_LIMIT,
    FilesystemMiddleware,
    _consecutive_execute_repeats,
)


class _RecordingSandbox(SandboxBackendProtocol, StateBackend):
    """Sandbox backend that records every command passed to `execute`."""

    def __init__(self) -> None:
        super().__init__()
        self.commands: list[str] = []

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.commands.append(command)
        return ExecuteResponse(output="", exit_code=124, truncated=False)

    @property
    def id(self) -> str:
        return "recording_sandbox"


def _execute_tool(backend: _RecordingSandbox):
    middleware = FilesystemMiddleware(backend=backend)
    return next(t for t in middleware.tools if t.name == "execute")


def _ai_execute(command: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "execute", "args": {"command": command}, "id": "tc"}],
    )


def _runtime(messages: list) -> ToolRuntime:
    return ToolRuntime(
        state={"messages": messages},
        context=None,
        tool_call_id="call-1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def test_counts_consecutive_identical_commands() -> None:
    messages = [_ai_execute("pytest"), _ai_execute("pytest")]
    assert _consecutive_execute_repeats(messages, "pytest") == 3


def test_normalizes_whitespace_when_counting() -> None:
    messages = [_ai_execute("pytest  tests"), _ai_execute("pytest tests")]
    assert _consecutive_execute_repeats(messages, "pytest   tests") == 3


def test_different_command_breaks_the_streak() -> None:
    messages = [_ai_execute("pytest"), _ai_execute("ls")]
    assert _consecutive_execute_repeats(messages, "ls") == 2


def test_guard_short_circuits_on_third_identical_call() -> None:
    backend = _RecordingSandbox()
    tool = _execute_tool(backend)
    messages = [_ai_execute("pytest"), _ai_execute("pytest")]

    result = tool.invoke({"command": "pytest", "runtime": _runtime(messages)})

    assert result.status == "error"
    assert "adapt" in result.content.lower() or "change your approach" in result.content.lower()
    # Backend was never invoked -- the guard short-circuited before execution.
    assert backend.commands == []


def test_guard_allows_first_two_attempts() -> None:
    backend = _RecordingSandbox()
    tool = _execute_tool(backend)
    messages = [_ai_execute("pytest")]

    result = tool.invoke({"command": "pytest", "runtime": _runtime(messages)})

    assert backend.commands == ["pytest"]
    assert result.status != "error" or "already been run" not in result.content


def test_limit_is_three() -> None:
    assert _EXECUTE_REPEAT_LIMIT == 3
