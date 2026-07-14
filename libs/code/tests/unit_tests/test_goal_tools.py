"""Unit tests for goal tools middleware."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import get_type_hints

import pytest
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from deepagents_code.goal_tools import (
    GOAL_TOOLS_SYSTEM_PROMPT,
    GoalToolsMiddleware,
    GoalToolState,
    _goal_snapshot,
    _goal_tool_state_context,
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


@pytest.mark.parametrize("status", ["paused", "complete"])
def test_rubric_snapshot_suppresses_inactive_goal_rubric(status: str) -> None:
    """Paused and completed goal criteria must not drive later work."""
    assert _rubric_snapshot(
        {
            "_goal_objective": "ship it",
            "_goal_status": status,
            "_goal_rubric": "- tests pass",
            "_sticky_rubric": "- tests pass",
        }
    ) == {
        "active": False,
        "criteria": None,
        "source": None,
        "grading_status": None,
    }


def test_rubric_snapshot_keeps_standalone_sticky_rubric_for_paused_goal() -> None:
    """An unrelated sticky rubric remains active while a goal is paused."""
    assert _rubric_snapshot(
        {
            "_goal_objective": "ship it",
            "_goal_status": "paused",
            "_goal_rubric": "- goal criteria",
            "_sticky_rubric": "- standalone criteria",
        }
    ) == {
        "active": True,
        "criteria": "- standalone criteria",
        "source": "sticky",
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


def test_goal_snapshot_paused_goal_is_inactive_but_persisted() -> None:
    """A paused goal remains readable without being actionable."""
    snapshot = _goal_snapshot(
        {
            "_goal_objective": "add refresh tokens",
            "_goal_status": "paused",
            "_goal_rubric": "- tests pass",
        }
    )

    assert snapshot == {
        "active": False,
        "objective": "add refresh tokens",
        "status": "paused",
        "criteria": "- tests pass",
        "note": None,
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


def test_goal_tool_state_context_exposes_private_active_state() -> None:
    """The model should see persisted activity even when chat history does not."""
    context = _goal_tool_state_context(
        {
            "_goal_objective": "add refresh tokens",
            "_goal_status": "active",
            "_goal_rubric": "- tests pass",
        }
    )

    assert "Goal status: `active`" in context
    assert "Goal actionable: `yes`" in context
    assert "Rubric active: `yes`" in context


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


def test_update_goal_requests_complete_with_note() -> None:
    """Completion is staged until the post-turn rubric result is available."""
    command = _update_goal_command(
        status="complete",
        note="tests pass",
        tool_call_id="call-1",
        state={"_goal_objective": "add refresh tokens"},
    )

    assert isinstance(command, Command)
    assert command.update is not None
    assert command.update["_pending_goal_completion_note"] == "tests pass"
    assert "_goal_status" not in command.update
    assert "_goal_status_note" not in command.update
    message = command.update["messages"][0]
    assert message.content == (
        "Goal completion requested. It will be recorded if the accepted rubric "
        "is satisfied."
    )
    assert message.tool_call_id == "call-1"


@pytest.mark.parametrize("rubric_status", [None, "needs_revision", "satisfied"])
def test_update_goal_completion_request_ignores_current_rubric_status(
    rubric_status: str | None,
) -> None:
    """The final rubric result is checked after the agent turn, not in-tool."""
    state = {"_goal_objective": "add refresh tokens"}
    if rubric_status is not None:
        state["_rubric_status"] = rubric_status
    command = _update_goal_command(
        status="complete",
        note="tests pass",
        tool_call_id="call-1",
        state=state,
    )

    assert isinstance(command, Command)
    assert command.update is not None
    assert command.update["_pending_goal_completion_note"] == "tests pass"


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
    assert command.update["_pending_goal_completion_note"] is None
    assert (
        command.update["messages"][0].content
        == "Goal marked blocked. waiting on API docs"
    )


def test_update_goal_rejects_status_change_while_paused() -> None:
    """The model cannot resume or complete a user-paused goal."""
    command = _update_goal_command(
        status="complete",
        note="tests pass",
        tool_call_id="call-1",
        state={
            "_goal_objective": "add refresh tokens",
            "_goal_status": "paused",
        },
    )

    assert command.update is not None
    assert set(command.update) == {"messages"}
    assert "`/goal resume`" in command.update["messages"][0].content


def test_update_goal_rejects_status_change_after_completion() -> None:
    """A completed goal remains terminal on later agent turns."""
    command = _update_goal_command(
        status="blocked",
        note="new blocker",
        tool_call_id="call-1",
        state={
            "_goal_objective": "add refresh tokens",
            "_goal_status": "complete",
        },
    )

    assert command.update is not None
    assert set(command.update) == {"messages"}
    assert "already complete" in command.update["messages"][0].content


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
    assert command.update["_pending_goal_completion_note"] == "all green"
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
    state: dict[str, object] | None = None,
) -> SimpleNamespace:
    """Build a `ModelRequest`-shaped double with an `override` that mirrors it."""
    return SimpleNamespace(
        system_message=system_message,
        runtime=SimpleNamespace(context=context or {}),
        state=state or {},
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
    assert blocks[-1]["text"].strip().startswith(GOAL_TOOLS_SYSTEM_PROMPT)
    assert "Goal status: `not set`" in blocks[-1]["text"]


def test_wrap_model_call_seeds_guidance_without_system_message() -> None:
    """Guidance should seed a fresh system message when none exists."""
    captured: dict[str, SimpleNamespace] = {}
    request = _fake_request(None)

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    new_system = captured["request"].system_message
    text = new_system.content[0]["text"]
    assert text.startswith(GOAL_TOOLS_SYSTEM_PROMPT)
    assert "Goal actionable: `no`" in text
    assert "Rubric active: `no`" in text


def test_wrap_model_call_exposes_active_private_state() -> None:
    """Private goal channels should control model-visible tool gating."""
    captured: dict[str, SimpleNamespace] = {}
    request = _fake_request(
        None,
        state={
            "_goal_objective": "ship it",
            "_goal_status": "blocked",
            "_goal_rubric": "- tests pass",
        },
    )

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    text = captured["request"].system_message.content[0]["text"]
    assert "Goal status: `blocked`" in text
    assert "Goal actionable: `yes`" in text
    assert "Rubric active: `yes`" in text


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
    assert blocks[-1]["text"].strip().startswith(GOAL_TOOLS_SYSTEM_PROMPT)


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
        "_pending_goal_completion_note",
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
