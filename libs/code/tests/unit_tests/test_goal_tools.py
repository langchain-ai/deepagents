"""Unit tests for goal tools middleware."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import get_type_hints

from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from deepagents_code.goal_tools import (
    GOAL_TOOLS_SYSTEM_PROMPT,
    GoalToolsMiddleware,
    GoalToolState,
    _goal_snapshot,
    _rubric_snapshot,
    _update_goal_command,
)


def test_rubric_snapshot_without_rubric() -> None:
    """`get_rubric` should report inactive state when no criteria are set."""
    assert _rubric_snapshot({}) == {
        "active": False,
        "criteria": None,
        "source": None,
        "grading_status": None,
    }


def test_rubric_snapshot_prefers_current_invocation_rubric() -> None:
    """The public `rubric` state is what `RubricMiddleware` grades this turn."""
    assert _rubric_snapshot(
        {
            "rubric": "- one-shot criteria",
            "_sticky_rubric": "- sticky criteria",
            "_goal_objective": "ship it",
            "_goal_rubric": "- goal criteria",
            "_rubric_status": "needs_revision",
        }
    ) == {
        "active": True,
        "criteria": "- one-shot criteria",
        "source": "invocation",
        "grading_status": "needs_revision",
    }


def test_rubric_snapshot_identifies_goal_rubric() -> None:
    """A goal-backed rubric should be labeled as goal criteria."""
    assert _rubric_snapshot(
        {
            "rubric": "- tests pass",
            "_goal_objective": "ship it",
            "_goal_rubric": "- tests pass",
        }
    ) == {
        "active": True,
        "criteria": "- tests pass",
        "source": "goal",
        "grading_status": None,
    }


def test_rubric_snapshot_restores_sticky_rubric_without_public_input() -> None:
    """Persisted sticky rubric should be visible outside an active turn."""
    assert _rubric_snapshot({"_sticky_rubric": "- sticky criteria"}) == {
        "active": True,
        "criteria": "- sticky criteria",
        "source": "sticky",
        "grading_status": None,
    }


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


def test_goal_snapshot_complete_goal_is_inactive() -> None:
    """A completed goal must report `active=False` (status drives the flag)."""
    snapshot = _goal_snapshot(
        {
            "_goal_objective": "add refresh tokens",
            "_goal_status": "complete",
            "_goal_status_note": "tests pass",
        }
    )
    assert snapshot["active"] is False
    assert snapshot["status"] == "complete"


def test_goal_snapshot_objective_without_status_defaults_active() -> None:
    """An objective with no recorded status reads as active, not contradictory."""
    snapshot = _goal_snapshot({"_goal_objective": "add refresh tokens"})
    assert snapshot["active"] is True
    assert snapshot["status"] == "active"


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


def test_update_goal_marks_blocked_with_note() -> None:
    """`update_goal` should record a blocker plus its evidence."""
    command = _update_goal_command(
        status="blocked",
        note="waiting on API docs",
        tool_call_id="call-1",
        state={"_goal_objective": "add refresh tokens"},
    )

    assert isinstance(command, Command)
    assert command.update is not None
    assert command.update["_goal_status"] == "blocked"
    assert command.update["_goal_status_note"] == "waiting on API docs"
    assert (
        command.update["messages"][0].content
        == "Goal marked blocked. waiting on API docs"
    )


def test_update_goal_rejects_empty_note() -> None:
    """Evidence is required: an empty note must not commit a status."""
    command = _update_goal_command(
        status="complete",
        note="   ",
        tool_call_id="call-1",
        state={"_goal_objective": "add refresh tokens"},
    )

    assert isinstance(command, Command)
    assert command.update is not None
    assert set(command.update) == {"messages"}
    message = command.update["messages"][0]
    assert "evidence" in message.content
    assert message.tool_call_id == "call-1"


def test_get_rubric_tool_invokes_snapshot() -> None:
    """The registered `get_rubric` tool should delegate to `_rubric_snapshot`."""
    middleware = GoalToolsMiddleware()
    get_rubric = next(t for t in middleware.tools if t.name == "get_rubric")
    result = get_rubric.func(  # ty: ignore[unresolved-attribute]
        state={"rubric": "- tests pass"}
    )
    assert result["criteria"] == "- tests pass"
    assert result["active"] is True


def test_get_goal_tool_invokes_snapshot() -> None:
    """The registered `get_goal` tool should delegate to `_goal_snapshot`."""
    middleware = GoalToolsMiddleware()
    get_goal = next(t for t in middleware.tools if t.name == "get_goal")
    result = get_goal.func(  # ty: ignore[unresolved-attribute]
        state={"_goal_objective": "ship it", "_goal_status": "active"}
    )
    assert result["objective"] == "ship it"
    assert result["active"] is True


def test_update_goal_tool_invokes_command_builder() -> None:
    """The registered `update_goal` tool should wire all args to the helper."""
    middleware = GoalToolsMiddleware()
    update_goal = next(t for t in middleware.tools if t.name == "update_goal")
    command = update_goal.func(  # ty: ignore[unresolved-attribute]
        status="complete",
        note="all green",
        tool_call_id="call-9",
        state={"_goal_objective": "ship it"},
    )
    assert isinstance(command, Command)
    assert command.update is not None
    assert command.update["_goal_status"] == "complete"
    assert command.update["messages"][0].tool_call_id == "call-9"


def _capturing_handler(
    captured: dict[str, SimpleNamespace],
) -> Callable[[SimpleNamespace], str]:
    """Build a sync handler that records the request it receives."""

    def handler(request: SimpleNamespace) -> str:
        captured["request"] = request
        return "response"

    return handler


def _fake_request(
    system_message: SystemMessage | None,
    *,
    context: object | None = None,
) -> SimpleNamespace:
    """Build a `ModelRequest`-shaped double with an `override` that mirrors it."""
    return SimpleNamespace(
        system_message=system_message,
        runtime=SimpleNamespace(context=context or {}),
        override=lambda **kw: SimpleNamespace(**kw),
    )


def test_wrap_model_call_appends_guidance_to_existing_prompt() -> None:
    """Guidance should append to an existing system message's blocks."""
    captured: dict[str, SimpleNamespace] = {}
    request = _fake_request(SystemMessage(content="base instructions"))

    result = GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    assert result == "response"
    new_system = captured["request"].system_message
    assert isinstance(new_system, SystemMessage)
    blocks = new_system.content
    assert blocks[0]["text"] == "base instructions"
    assert blocks[-1]["text"].strip() == GOAL_TOOLS_SYSTEM_PROMPT


def test_wrap_model_call_seeds_guidance_without_system_message() -> None:
    """Guidance should seed a fresh system message when none exists."""
    captured: dict[str, SimpleNamespace] = {}
    request = _fake_request(None)

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    new_system = captured["request"].system_message
    assert new_system.content == [{"type": "text", "text": GOAL_TOOLS_SYSTEM_PROMPT}]


def test_wrap_model_call_appends_blocked_goal_retry_context() -> None:
    """Retry context should reach the model through runtime context."""
    captured: dict[str, SimpleNamespace] = {}
    request = _fake_request(
        None,
        context={"blocked_goal_retry_context": "<dcode_blocked_goal_retry_context />"},
    )

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    new_system = captured["request"].system_message
    text = new_system.content[0]["text"]
    assert GOAL_TOOLS_SYSTEM_PROMPT in text
    assert "<dcode_blocked_goal_retry_context />" in text


async def test_awrap_model_call_appends_guidance_to_existing_prompt() -> None:
    """The async path should mirror the sync guidance injection."""
    captured: dict[str, SimpleNamespace] = {}

    async def handler(request: SimpleNamespace) -> str:  # noqa: RUF029
        captured["request"] = request
        return "response"

    request = _fake_request(SystemMessage(content="base instructions"))

    result = await GoalToolsMiddleware().awrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        handler,  # ty: ignore[invalid-argument-type]
    )

    assert result == "response"
    blocks = captured["request"].system_message.content
    assert blocks[0]["text"] == "base instructions"
    assert blocks[-1]["text"].strip() == GOAL_TOOLS_SYSTEM_PROMPT


def test_goal_tool_state_marks_goal_fields_private() -> None:
    """`_goal_*` channels must stay private so they don't leak into the schema.

    The channels are inherited from `GoalRubricChannels`. Resolving the full
    hints the way LangGraph does (`get_type_hints(..., include_extras=True)`,
    which walks the MRO) confirms the `PrivateStateAttr` markers carry through
    inheritance, while the public `rubric` input stays non-private.
    """
    hints = get_type_hints(GoalToolState, include_extras=True)
    for field in (
        "_goal_objective",
        "_goal_status",
        "_goal_rubric",
        "_goal_status_note",
        "_sticky_rubric",
    ):
        assert PrivateStateAttr in getattr(hints[field], "__metadata__", ())
    # `rubric` is the public `RubricMiddleware` input and stays non-private.
    assert PrivateStateAttr not in getattr(hints["rubric"], "__metadata__", ())


def test_goal_tools_middleware_registers_tools() -> None:
    """Middleware should expose exactly the constrained rubric and goal tools."""
    middleware = GoalToolsMiddleware()
    assert [tool.name for tool in middleware.tools] == [
        "get_rubric",
        "get_goal",
        "update_goal",
    ]
