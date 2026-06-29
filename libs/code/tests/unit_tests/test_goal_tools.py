"""Unit tests for goal tools middleware."""

from langgraph.types import Command

from deepagents_code.goal_tools import (
    GoalToolsMiddleware,
    _goal_snapshot,
    _update_goal_command,
)


def test_goal_snapshot_without_goal_preserves_rubric() -> None:
    """`get_goal` should report inactive state while still showing criteria."""
    assert _goal_snapshot({"rubric": "- tests pass"}) == {
        "active": False,
        "objective": None,
        "status": None,
        "criteria": "- tests pass",
        "note": None,
    }


def test_goal_snapshot_with_active_goal() -> None:
    """`get_goal` should expose objective, status, criteria, and note."""
    assert _goal_snapshot(
        {
            "_goal_objective": "add refresh tokens",
            "_goal_status": "blocked",
            "_goal_rubric": "- tests pass",
            "_goal_status_note": "waiting on API docs",
        }
    ) == {
        "active": True,
        "objective": "add refresh tokens",
        "status": "blocked",
        "criteria": "- tests pass",
        "note": "waiting on API docs",
    }


def test_update_goal_without_active_goal_returns_tool_message_only() -> None:
    """`update_goal` should not invent goals when none exists."""
    command = _update_goal_command(
        status="complete",
        note="done",
        tool_call_id="call-1",
        state={},
    )

    assert isinstance(command, Command)
    assert command.update is not None
    assert set(command.update) == {"messages"}
    message = command.update["messages"][0]
    assert message.content == "No active goal is set."
    assert message.tool_call_id == "call-1"


def test_update_goal_marks_complete_with_note() -> None:
    """`update_goal` should only update status/note plus tool response."""
    command = _update_goal_command(
        status="complete",
        note="tests pass",
        tool_call_id="call-1",
        state={"_goal_objective": "add refresh tokens"},
    )

    assert isinstance(command, Command)
    assert command.update is not None
    assert command.update["_goal_status"] == "complete"
    assert command.update["_goal_status_note"] == "tests pass"
    message = command.update["messages"][0]
    assert message.content == "Goal marked complete. tests pass"
    assert message.tool_call_id == "call-1"


def test_goal_tools_middleware_registers_tools() -> None:
    """Middleware should expose exactly the constrained goal tools."""
    middleware = GoalToolsMiddleware()
    assert [tool.name for tool in middleware.tools] == ["get_goal", "update_goal"]
