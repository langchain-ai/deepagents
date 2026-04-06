"""Unit tests for tau3 multi-turn runner behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from tests.evals.tau3_rhobank.runner import run_multi_turn
from tests.evals.tau3_rhobank.user_sim import UserResponse


class _FakeTool:
    def __init__(self, name: str, calls: list[tuple[str, dict[str, object]]]) -> None:
        self.name = name
        self._calls = calls

    def invoke(self, args: dict[str, object]) -> str:
        self._calls.append((self.name, args))
        return f"{self.name}-ok"


class _FakeUserSim:
    def __init__(self) -> None:
        self._tool_results_seen = 0
        self.received_tool_results: list[tuple[str, str, str]] = []

    @property
    def is_done(self) -> bool:
        return False

    def get_opening_message(self) -> str:
        return "opening message"

    def respond(self, agent_message: str) -> UserResponse:
        return UserResponse(
            tool_calls=[{"id": "tc1", "name": "first_tool", "args": {"a": 1}}],
        )

    def receive_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> None:
        self._tool_results_seen += 1
        self.received_tool_results.append((tool_call_id, tool_name, result))

    def get_response_after_tools(self) -> UserResponse:
        if self._tool_results_seen == 1:
            return UserResponse(
                tool_calls=[{"id": "tc2", "name": "second_tool", "args": {"b": 2}}],
            )
        return UserResponse(text="final user text")


def test_run_multi_turn_executes_chained_user_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner executes user tool calls until text is returned."""
    monkeypatch.setattr(
        "tests.evals.tau3_rhobank.runner.run_agent",
        lambda *_args, **_kwargs: SimpleNamespace(answer="assistant reply"),
    )

    calls: list[tuple[str, dict[str, object]]] = []
    tools = [
        _FakeTool("first_tool", calls),
        _FakeTool("second_tool", calls),
    ]
    user_sim = _FakeUserSim()

    result = run_multi_turn(
        agent=object(),
        user_sim=user_sim,
        model=object(),
        tool_call_log=[],
        user_tools=tools,
        max_turns=1,
    )

    assert calls == [
        ("first_tool", {"a": 1}),
        ("second_tool", {"b": 2}),
    ]
    assert user_sim.received_tool_results == [
        ("tc1", "first_tool", "first_tool-ok"),
        ("tc2", "second_tool", "second_tool-ok"),
    ]
    assert result.messages[-1].role == "user"
    assert result.messages[-1].content == "final user text"
