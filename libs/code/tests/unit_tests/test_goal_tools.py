"""Unit tests for goal tools middleware."""

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest
from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command

from deepagents_code.goal_state_notice import (
    build_goal_continuation,
    build_goal_state_notice,
    goal_state_notice_info,
)
from deepagents_code.goal_tools import (
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
    messages = command.update["messages"]
    assert len(messages) == 1
    assert messages[0].content == "Goal marked blocked. waiting on API docs"


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
    messages: list[object] | None = None,
) -> SimpleNamespace:
    """Build a `ModelRequest`-shaped double with an `override` that mirrors it."""
    request = SimpleNamespace(
        system_message=system_message,
        runtime=SimpleNamespace(context=context or {}),
        state=state or {},
        messages=messages or [],
    )

    def override(**kw: object) -> SimpleNamespace:
        updated = SimpleNamespace(**vars(request))
        updated.__dict__.update(kw)
        return updated

    request.override = override
    return request


def test_before_model_persists_public_rubric_notice() -> None:
    state = cast(
        "AgentState[Any]",
        {
            "rubric": "include a marker",
            "messages": [HumanMessage(content="answer the question")],
        },
    )

    update = GoalToolsMiddleware._notice_update(state)

    assert update is not None
    notice = update["messages"][0]
    assert "Rubric active: yes" in notice.content
    assert goal_state_notice_info(notice) is not None


def test_before_model_appends_blocked_notice_after_parallel_tool_results() -> None:
    assistant = AIMessage(
        content="",
        tool_calls=[
            {"name": "update_goal", "args": {}, "id": "goal-call"},
            {"name": "other_tool", "args": {}, "id": "other-call"},
        ],
    )
    state = cast(
        "AgentState[Any]",
        {
            "_goal_objective": "ship it",
            "_goal_status": "blocked",
            "_goal_status_note": "waiting",
            "messages": [
                assistant,
                ToolMessage(content="blocked", tool_call_id="goal-call"),
                ToolMessage(content="done", tool_call_id="other-call"),
            ],
        },
    )

    update = GoalToolsMiddleware._notice_update(state)

    assert update is not None
    combined = [*state["messages"], *update["messages"]]
    assert isinstance(combined[-2], ToolMessage)
    assert isinstance(combined[-1], HumanMessage)
    assert "Goal status: blocked" in combined[-1].content


def test_notice_update_is_none_when_current_notice_already_present() -> None:
    # Idempotence at the layer where a double-append would occur: once
    # `before_model` has persisted the current notice, a second boundary must
    # not append another copy.
    goal_state = {
        "_goal_objective": "ship it",
        "_goal_status": "active",
        "_goal_rubric": "tests pass",
    }
    notice = build_goal_state_notice(goal_state)
    state = cast(
        "AgentState[Any]",
        {**goal_state, "messages": [HumanMessage(content="go"), notice]},
    )

    assert GoalToolsMiddleware._notice_update(state) is None


def test_notice_update_is_none_for_empty_state() -> None:
    state = cast(
        "AgentState[Any]",
        {"messages": [HumanMessage(content="just chatting")]},
    )

    assert GoalToolsMiddleware._notice_update(state) is None


async def test_abefore_model_matches_before_model() -> None:
    # The async boundary must produce the same notice update as the sync one;
    # tests elsewhere only exercise `_notice_update` directly, so drive the
    # overrides themselves here.
    goal_state = {
        "rubric": "include a marker",
        "messages": [HumanMessage(content="answer the question")],
    }
    sync_state = cast("AgentState[Any]", dict(goal_state))
    async_state = cast("AgentState[Any]", dict(goal_state))
    middleware = GoalToolsMiddleware()
    runtime = cast("Any", SimpleNamespace(context={}))

    sync_update = middleware.before_model(sync_state, runtime)
    async_update = await middleware.abefore_model(async_state, runtime)

    assert sync_update is not None
    assert async_update is not None
    sync_notice = sync_update["messages"][0]
    async_notice = async_update["messages"][0]
    assert "Rubric active: yes" in sync_notice.content
    assert async_notice.content == sync_notice.content
    assert (
        async_notice.additional_kwargs["state_fingerprint"]
        == sync_notice.additional_kwargs["state_fingerprint"]
    )


def test_wrap_model_call_restores_notice_after_compaction() -> None:
    state: dict[str, object] = {
        "_goal_objective": "ship it",
        "_goal_status": "active",
        "_goal_rubric": "tests pass",
    }
    request = _fake_request(
        None,
        state=state,
        messages=[HumanMessage(content="continue")],
    )
    captured: dict[str, SimpleNamespace] = {}

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    notice = captured["request"].messages[-1]
    assert "Goal status: active" in notice.content
    assert goal_state_notice_info(notice) is not None


def test_wrap_model_call_does_not_restore_stale_state_over_unsaved_fallback() -> None:
    state: dict[str, object] = {
        "_goal_objective": "old goal",
        "_goal_status": "active",
        "_goal_rubric": "old rubric",
    }
    fallback = build_goal_continuation(
        "created",
        unsaved_objective="new unsaved goal",
    )
    request = _fake_request(None, state=state, messages=[fallback])
    captured: dict[str, SimpleNamespace] = {}

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    assert captured["request"].messages == [fallback]


def test_wrap_model_call_leaves_system_message_unchanged() -> None:
    """Request wrapping must not mutate the system prompt."""
    system = SystemMessage(content="base instructions")
    captured: dict[str, SimpleNamespace] = {}
    request = _fake_request(system)

    result = GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    assert result == "response"
    assert captured["request"] is request
    assert captured["request"].system_message is system


def test_wrap_model_call_leaves_missing_system_message_none() -> None:
    """Request wrapping must not invent a system message when none exists."""
    captured: dict[str, SimpleNamespace] = {}
    request = _fake_request(None)

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    assert captured["request"] is request
    assert captured["request"].system_message is None


def test_system_prompt_and_tool_schemas_are_byte_stable_across_states() -> None:
    """Goal lifecycle state must not change cache-sensitive request prefixes."""
    base_system = SystemMessage(content="base instructions")
    states: list[dict[str, object]] = [
        {},
        {
            "_goal_objective": "ship it",
            "_goal_status": "active",
            "_goal_rubric": "tests pass",
        },
        {
            "_goal_objective": "ship it",
            "_goal_status": "blocked",
            "_goal_status_note": "waiting",
            "_goal_rubric": "tests pass",
        },
        {
            "_goal_objective": "ship it",
            "_goal_status": "paused",
            "_goal_rubric": "tests pass",
        },
        {
            "_goal_objective": "ship it",
            "_goal_status": "complete",
            "_goal_rubric": "tests pass",
        },
        {
            "rubric": None,
            "_sticky_rubric": None,
            "_goal_objective": None,
            "_goal_status": None,
            "_goal_rubric": None,
            "_goal_status_note": None,
        },
    ]
    system_refs: list[object] = []
    schema_bytes: list[bytes] = []

    for state in states:
        captured: dict[str, SimpleNamespace] = {}
        middleware = GoalToolsMiddleware()
        request = _fake_request(base_system, state=state)
        middleware.wrap_model_call(
            request,  # ty: ignore[invalid-argument-type]
            _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
        )
        system_refs.append(captured["request"].system_message)
        schemas = [convert_to_openai_tool(tool) for tool in middleware.tools]
        schema_bytes.append(
            json.dumps(schemas, sort_keys=True, separators=(",", ":")).encode()
        )

    assert all(system is base_system for system in system_refs)
    assert len(set(schema_bytes)) == 1
    assert b"Current Persisted Goal/Rubric State" not in schema_bytes[0]
    assert b"blocked_goal_retry_context" not in schema_bytes[0]


async def test_awrap_model_call_leaves_system_message_unchanged() -> None:
    """The async path should also leave the system prompt alone."""
    system = SystemMessage(content="base instructions")
    captured: dict[str, SimpleNamespace] = {}

    async def handler(request: SimpleNamespace) -> str:  # noqa: RUF029
        captured["request"] = request
        return "response"

    request = _fake_request(system)

    result = await GoalToolsMiddleware().awrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        handler,  # ty: ignore[invalid-argument-type]
    )

    assert result == "response"
    assert captured["request"] is request
    assert captured["request"].system_message is system


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
