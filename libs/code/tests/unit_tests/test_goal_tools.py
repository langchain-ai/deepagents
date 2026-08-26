"""Unit tests for goal tools middleware."""

import json
import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, NamedTuple, cast, get_type_hints

import pytest
from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command

from deepagents_code.goal_state_limits import (
    GOAL_STATUS_NOTE_CHAR_LIMIT,
    RUBRIC_CHAR_LIMIT,
)
from deepagents_code.goal_state_notice import (
    GOAL_MESSAGE_SCHEMA_VERSION,
    build_goal_continuation,
    build_goal_state_notice,
    goal_state_fingerprint,
    goal_state_notice_info,
)
from deepagents_code.goal_tools import (
    GoalToolsMiddleware,
    GoalToolState,
    _update_goal_command,
)


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


def test_update_goal_rejects_oversized_note_without_changing_status() -> None:
    """Runtime validation covers calls that bypass the generated tool schema."""
    command = _update_goal_command(
        status="blocked",
        note="x" * (GOAL_STATUS_NOTE_CHAR_LIMIT + 1),
        tool_call_id="call-1",
        state={"_goal_objective": "ship it", "_goal_status": "active"},
    )

    assert command.update is not None
    assert "_goal_status" not in command.update
    message = command.update["messages"][0]
    assert "maximum is 4,000" in message.content


def test_update_goal_refuses_when_saved_state_is_too_large() -> None:
    """The notice says the goal is unusable, so the write side must agree.

    `update_goal` is the only goal surface left after the read tools were
    removed, so "do not work toward it and do not grade against it" cannot rest
    on notice prose alone.
    """
    command = _update_goal_command(
        status="blocked",
        note="waiting on docs",
        tool_call_id="call-1",
        state={
            "_goal_objective": "ship it",
            "_goal_status": "active",
            "_goal_rubric": "x" * (RUBRIC_CHAR_LIMIT + 1),
        },
    )

    assert command.update is not None
    assert set(command.update) == {"messages"}
    content = command.update["messages"][0].content
    assert "too large to use" in content
    assert "clear and recreate the goal" in content
    assert "Rubric is" in content


def test_update_goal_refuses_when_a_legacy_status_note_is_too_large() -> None:
    """The model cannot shorten its own oversized note; only the user can.

    An actionable goal embeds `_goal_status_note` in its notice, so a legacy
    checkpoint carrying an oversized one trips the same guard as an oversized
    rubric — and `update_goal` is the only surface the model has to replace it.
    Recovery is therefore user-only, via `/goal clear`. Every other oversized test
    triggers on the objective or the rubric, so this pins the one case where the
    model's own prior write is what locks it out.
    """
    command = _update_goal_command(
        status="blocked",
        note="a short new blocker",
        tool_call_id="call-1",
        state={
            "_goal_objective": "ship it",
            "_goal_status": "active",
            "_goal_status_note": "x" * (GOAL_STATUS_NOTE_CHAR_LIMIT + 1),
        },
    )

    assert command.update is not None
    assert set(command.update) == {"messages"}
    content = command.update["messages"][0].content
    assert "too large to use" in content
    # The message must point at the user action, since no model action can fix it.
    assert "clear and recreate the goal" in content
    assert "Goal status note is" in content


def test_update_goal_commits_when_state_fits_the_notice_budget() -> None:
    """The oversized guard must not block a normal blocker report."""
    command = _update_goal_command(
        status="blocked",
        note="waiting on docs",
        tool_call_id="call-1",
        state={
            "_goal_objective": "ship it",
            "_goal_status": "active",
            "_goal_rubric": "tests pass",
        },
    )

    assert command.update is not None
    assert command.update["_goal_status"] == "blocked"


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
    """Build a `ModelRequest`-shaped double with an `override` that mirrors it.

    `override` rebuilds from the receiver rather than from this factory's
    request, so chained calls compose the way `ModelRequest.override` does.
    Rebuilding from the original would silently drop an earlier override — for
    example the `_summarization_event` reset applied before a notice is pinned.
    """
    request = SimpleNamespace(
        system_message=system_message,
        runtime=SimpleNamespace(context=context or {}),
        state=state or {},
        messages=messages or [],
    )

    def _attach(target: SimpleNamespace) -> SimpleNamespace:
        def override(**kw: object) -> SimpleNamespace:
            updated = SimpleNamespace(**vars(target))
            updated.__dict__.update(kw)
            return _attach(updated)

        target.override = override
        return target

    return _attach(request)


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


def test_notice_update_disables_grading_for_oversized_legacy_rubric() -> None:
    """A bounded fallback notice also suppresses the grader's rubric input."""
    rubric = "x" * (RUBRIC_CHAR_LIMIT + 1)
    state = cast(
        "AgentState[Any]",
        {
            "rubric": rubric,
            "_sticky_rubric": rubric,
            "messages": [HumanMessage(content="continue")],
        },
    )

    update = GoalToolsMiddleware._notice_update(state)

    assert update is not None
    assert update["rubric"] is None
    assert "too large to include safely" in update["messages"][0].content


def test_oversized_state_notice_settles_after_one_append() -> None:
    """The bounded fallback must not re-append on every later boundary.

    `messages` is an append-only channel, so a predicate that never considers
    itself satisfied grows history once per turn for the rest of the thread. This
    converges today only because the fallback fingerprints the *state* rather
    than its own bounded content — a refactor to fingerprint the rendered text
    would turn the fallback into an unbounded per-turn append.
    """
    rubric = "x" * (RUBRIC_CHAR_LIMIT + 1)
    messages: list[object] = [HumanMessage(content="continue")]
    state = cast(
        "AgentState[Any]",
        {"rubric": rubric, "_sticky_rubric": rubric, "messages": messages},
    )

    first = GoalToolsMiddleware._notice_update(state)

    assert first is not None
    # `rubric` is cleared alongside the notice, so the settled state reflects
    # both halves of the update the graph would apply.
    settled = cast(
        "AgentState[Any]",
        {
            "rubric": None,
            "_sticky_rubric": rubric,
            "messages": [*messages, *first["messages"]],
        },
    )
    for _ in range(2):
        assert GoalToolsMiddleware._notice_update(settled) is None


def test_oversized_state_keeps_clearing_the_rubric_on_later_turns() -> None:
    """The steady state is a `rubric` clear with no `messages` key at all.

    `rubric` is a public per-invocation input the app re-sets each turn, so from
    turn two onward the notice already matches and only the clear is left. If
    the clear were ever made conditional on writing a notice, grading would
    silently resume re-injecting the oversized criteria from turn two on — the
    exact failure the oversized path exists to prevent.
    """
    rubric = "x" * (RUBRIC_CHAR_LIMIT + 1)
    messages: list[object] = [HumanMessage(content="continue")]
    state = cast(
        "AgentState[Any]",
        {"rubric": rubric, "_sticky_rubric": rubric, "messages": messages},
    )

    first = GoalToolsMiddleware._notice_update(state)

    assert first is not None
    assert set(first) == {"messages", "rubric"}
    assert first["rubric"] is None

    # Turn two: the notice is in history, but the app has re-set `rubric`.
    resent = cast(
        "AgentState[Any]",
        {
            "rubric": rubric,
            "_sticky_rubric": rubric,
            "messages": [*messages, *first["messages"]],
        },
    )
    second = GoalToolsMiddleware._notice_update(resent)

    assert second is not None
    assert set(second) == {"rubric"}
    assert second["rubric"] is None


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


@pytest.mark.parametrize("cutoff", [-1, 99])
def test_wrap_model_call_disables_malformed_summarization_cutoff(
    cutoff: int,
) -> None:
    """An invalid restored cutoff cannot strip the only visible goal notice."""
    state: dict[str, object] = {
        "_goal_objective": "ship it",
        "_goal_status": "active",
        "_summarization_event": {
            "summary_message": HumanMessage(content="summary"),
            "cutoff_index": cutoff,
        },
    }
    notice = build_goal_state_notice(state, event_id="persisted")
    request = _fake_request(
        None,
        state=state,
        messages=[notice, HumanMessage(content="continue")],
    )
    captured: dict[str, SimpleNamespace] = {}

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    assert captured["request"].state["_summarization_event"] is None
    # Discarding the event forces a fresh tail notice without changing the
    # cacheable request prefix.
    sent = captured["request"].messages
    assert sent[:-1] == request.messages
    assert goal_state_notice_info(sent[-1]) is not None
    assert sum(goal_state_notice_info(m) is not None for m in sent) == 2
    assert state["_summarization_event"] is not None


def test_wrap_model_call_replaces_superseded_oversized_notice() -> None:
    """A bounded same-index stand-in keeps legacy poison from the model."""
    rubric = "x" * (RUBRIC_CHAR_LIMIT + 1)
    state: dict[str, object] = {
        "rubric": rubric,
        "_sticky_rubric": rubric,
    }
    legacy = build_goal_state_notice({"rubric": "old"}, event_id="legacy-oversized")
    legacy.content = (
        "legacy notice\n<acceptance_criteria>"
        f"{'x' * (RUBRIC_CHAR_LIMIT + 1)}"
        "</acceptance_criteria>"
    )
    legacy.additional_kwargs = {
        **legacy.additional_kwargs,
        "goal_message_schema_version": GOAL_MESSAGE_SCHEMA_VERSION - 1,
    }
    request = _fake_request(
        None,
        state=state,
        messages=[HumanMessage(content="continue"), legacy],
    )
    captured: dict[str, SimpleNamespace] = {}

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    messages = captured["request"].messages
    assert len(messages) == 3
    assert messages[0] is request.messages[0]
    assert messages[1].id == legacy.id
    assert "oversized superseded goal/rubric state notice was omitted" in (
        messages[1].content
    )
    assert rubric not in messages[1].content
    assert "too large to include safely" in messages[-1].content
    assert len(messages[-1].content) < 2_000


def test_wrap_model_call_preserves_latest_authoritative_oversized_notice() -> None:
    """The only current notice remains visible until a successor is appended."""
    state: dict[str, object] = {"rubric": "current"}
    superseded = build_goal_state_notice(
        {"rubric": "old"}, event_id="superseded-oversized"
    )
    authoritative = build_goal_state_notice(state, event_id="authoritative-oversized")
    oversized_content = (
        "notice\n<acceptance_criteria>"
        f"{'x' * (RUBRIC_CHAR_LIMIT + 1)}"
        "</acceptance_criteria>"
    )
    superseded.content = oversized_content
    authoritative.content = oversized_content
    request = _fake_request(None, state=state, messages=[superseded, authoritative])
    captured: dict[str, SimpleNamespace] = {}

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    messages = captured["request"].messages
    assert "oversized superseded goal/rubric state notice was omitted" in (
        messages[0].content
    )
    assert messages[1] is authoritative
    assert messages[1].content == oversized_content


def test_wrap_model_call_preserves_bounded_stale_notice() -> None:
    """Schema-invalid history remains byte-stable while it fits the budget."""
    state: dict[str, object] = {"rubric": "current"}
    stale = build_goal_state_notice({"rubric": "old"}, event_id="bounded-stale")
    stale.additional_kwargs = {
        **stale.additional_kwargs,
        "goal_message_schema_version": GOAL_MESSAGE_SCHEMA_VERSION - 1,
    }
    request = _fake_request(None, state=state, messages=[stale])
    captured: dict[str, SimpleNamespace] = {}

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    messages = captured["request"].messages
    assert messages[0] is stale
    assert messages[-1] is not stale
    assert "Acceptance criteria" in messages[0].content


def test_discarding_malformed_summarization_event_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dropping the event also drops the summary, so it must not be silent."""
    state: dict[str, object] = {
        "_goal_objective": "ship it",
        "_goal_status": "active",
        "_summarization_event": {"cutoff_index": 99, "summary_message": "SUMMARY"},
        "messages": [HumanMessage(content="m0")],
    }

    with caplog.at_level(logging.WARNING, logger="deepagents_code.goal_state_notice"):
        update = GoalToolsMiddleware()._notice_update(
            cast("AgentState[Any]", state),
        )

    assert update is not None
    assert update["_summarization_event"] is None
    assert "Discarding malformed `_summarization_event`" in caplog.text
    assert "cutoff_index=99" in caplog.text
    assert "re-sends the full history" in caplog.text


def test_wrap_model_call_preserves_history_with_summarization_cutoff() -> None:
    """Appending a current notice leaves the summarizer's history unchanged."""
    state: dict[str, object] = {
        "_goal_objective": "ship the fix",
        "_goal_status": "active",
    }
    superseded = build_goal_state_notice({"rubric": "stale"}, event_id="superseded")
    previous = build_goal_state_notice({"rubric": "old"}, event_id="previous")
    tool_call = AIMessage(
        content="calling",
        tool_calls=[{"name": "read_file", "args": {}, "id": "call-1"}],
    )
    tool_result = ToolMessage(content="file body", tool_call_id="call-1")
    messages: list[object] = [
        HumanMessage(content="m0"),
        superseded,
        HumanMessage(content="m2"),
        tool_call,
        tool_result,
        previous,
    ]
    cutoff = 3
    summary = HumanMessage(content="SUMMARY")
    state["_summarization_event"] = {
        "cutoff_index": cutoff,
        "summary_message": summary,
    }
    request = _fake_request(None, state=state, messages=messages)
    captured: dict[str, SimpleNamespace] = {}

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    inner = captured["request"].messages
    assert inner[:-1] == messages
    assert inner[1] is superseded
    assert tool_call in inner[cutoff:]
    assert tool_result in inner[cutoff:]
    assert goal_state_notice_info(inner[-1]) is not None
    assert inner[-1].additional_kwargs["state_fingerprint"] == (
        goal_state_fingerprint(state)
    )


def test_wrap_model_call_preserves_history_without_summarization_event() -> None:
    """A current tail notice leaves the whole cacheable history byte-stable."""
    state: dict[str, object] = {
        "_goal_objective": "ship the fix",
        "_goal_status": "active",
    }
    stale_one = build_goal_state_notice({"rubric": "stale one"}, event_id="stale-1")
    stale_two = build_goal_state_notice({"rubric": "stale two"}, event_id="stale-2")
    previous = build_goal_state_notice({"rubric": "old"}, event_id="previous")
    tool_call = AIMessage(
        content="calling",
        tool_calls=[{"name": "read_file", "args": {}, "id": "call-1"}],
    )
    tool_result = ToolMessage(content="file body", tool_call_id="call-1")
    messages: list[object] = [
        HumanMessage(content="m0"),
        stale_one,
        AIMessage(content="a1"),
        tool_call,
        stale_two,
        tool_result,
        previous,
    ]
    request = _fake_request(None, state=state, messages=messages)
    captured: dict[str, SimpleNamespace] = {}

    GoalToolsMiddleware().wrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
    )

    inner = captured["request"].messages
    assert inner[:-1] == messages
    assert inner[1] is stale_one
    assert inner[4] is stale_two
    assert inner[-2] is previous
    assert goal_state_notice_info(inner[-1]) is not None
    assert sum(goal_state_notice_info(m) is not None for m in inner) == 4


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


class _NoticeCase(NamedTuple):
    """A goal state paired with what its notice is expected to expose.

    Expectations are carried per case rather than recomputed from the state, so a
    regression in `project_goal_state`'s actionability rule cannot move the
    expectation and the behavior together and still pass.
    """

    label: str
    state: dict[str, object]
    expect_notice: bool
    expect_objective: bool
    expect_criteria: bool
    expect_status_note: bool = False


_NOTICE_CASES = [
    _NoticeCase("no state at all", {}, False, False, False),
    _NoticeCase(
        "active goal with rubric",
        {
            "_goal_objective": "ship it",
            "_goal_status": "active",
            "_goal_rubric": "tests pass",
        },
        True,
        True,
        True,
    ),
    _NoticeCase(
        "blocked goal stays actionable",
        {
            "_goal_objective": "ship it",
            "_goal_status": "blocked",
            "_goal_status_note": "waiting",
            "_goal_rubric": "tests pass",
        },
        True,
        True,
        True,
        expect_status_note=True,
    ),
    _NoticeCase(
        "paused goal withholds everything",
        {
            "_goal_objective": "ship it",
            "_goal_status": "paused",
            "_goal_rubric": "tests pass",
        },
        True,
        False,
        False,
    ),
    _NoticeCase(
        "complete goal withholds everything",
        {
            "_goal_objective": "ship it",
            "_goal_status": "complete",
            "_goal_rubric": "tests pass",
        },
        True,
        False,
        False,
    ),
    _NoticeCase(
        # A one-shot rubric outlives a paused goal (see `app.py`'s `_next_rubric`
        # handling), so criteria render while the objective stays withheld.
        "paused goal with a one-shot rubric",
        {
            "_goal_objective": "ship it",
            "_goal_status": "paused",
            "rubric": "tests pass",
        },
        True,
        False,
        True,
    ),
    _NoticeCase(
        "explicitly cleared state",
        {
            "rubric": None,
            "_sticky_rubric": None,
            "_goal_objective": None,
            "_goal_status": None,
            "_goal_rubric": None,
            "_goal_status_note": None,
        },
        False,
        False,
        False,
    ),
]


def test_system_prompt_and_tool_schemas_are_byte_stable_across_states() -> None:
    """Goal lifecycle state must not change cache-sensitive request prefixes."""
    base_system = SystemMessage(content="base instructions")
    system_refs: list[object] = []
    schema_bytes: list[bytes] = []
    notice_texts: list[str] = []

    for case in _NOTICE_CASES:
        captured: dict[str, SimpleNamespace] = {}
        middleware = GoalToolsMiddleware()
        request = _fake_request(base_system, state=case.state)
        middleware.wrap_model_call(
            request,  # ty: ignore[invalid-argument-type]
            _capturing_handler(captured),  # ty: ignore[invalid-argument-type]
        )
        system_refs.append(captured["request"].system_message)
        notice_texts.append(
            "".join(
                message.content
                for message in captured["request"].messages
                if isinstance(getattr(message, "content", None), str)
            )
        )
        schemas = [convert_to_openai_tool(tool) for tool in middleware.tools]
        schema_bytes.append(
            json.dumps(schemas, sort_keys=True, separators=(",", ":")).encode()
        )

    assert all(system is base_system for system in system_refs)
    assert len(set(schema_bytes)) == 1
    # The system prompt and tool schemas stay byte-stable across goal states (so
    # the prompt-cache prefix is unaffected); the appended goal-state notice is
    # the only model-visible surface that varies.
    for case, notice_text in zip(_NOTICE_CASES, notice_texts, strict=True):
        if not case.expect_notice:
            assert notice_text == "", case.label
            continue
        assert "Goal status:" in notice_text, case.label
        assert ("ship it" in notice_text) is case.expect_objective, case.label
        assert ("tests pass" in notice_text) is case.expect_criteria, case.label
        # The status note follows the objective: readable while actionable,
        # withheld otherwise.
        assert ("waiting" in notice_text) is case.expect_status_note, case.label


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


async def test_awrap_model_call_restores_notice_after_compaction() -> None:
    """The async path must re-pin the notice, not just pass the request through.

    Async twin of `test_wrap_model_call_restores_notice_after_compaction`; without
    it, dropping the re-pin from `awrap_model_call` goes unnoticed because the
    other async wrap test uses a request that needs no notice.
    """
    captured: dict[str, SimpleNamespace] = {}

    async def handler(request: SimpleNamespace) -> str:  # noqa: RUF029
        captured["request"] = request
        return "response"

    request = _fake_request(
        None,
        state={
            "_goal_objective": "ship it",
            "_goal_status": "active",
            "_goal_rubric": "tests pass",
        },
        messages=[HumanMessage(content="continue")],
    )

    await GoalToolsMiddleware().awrap_model_call(
        request,  # ty: ignore[invalid-argument-type]
        handler,  # ty: ignore[invalid-argument-type]
    )

    notice = captured["request"].messages[-1]
    assert "Goal status: active" in notice.content
    assert "<goal_objective>ship it</goal_objective>" in notice.content
    assert goal_state_notice_info(notice) is not None


def test_stale_schema_notice_is_superseded_then_settles() -> None:
    """A resumed thread holding a prior-schema notice gets a current one, once.

    Prior notices could truncate required text or instruct the model to call read
    tools that no longer exist, so a resumed thread must not keep treating one as
    authoritative. The appended notice must also converge: the notice channel is
    append-only, so a predicate that never settles would grow history every turn.
    """
    state: dict[str, object] = {
        "_goal_objective": "ship it",
        "_goal_status": "active",
        "_goal_rubric": "tests pass",
    }
    stale = build_goal_state_notice(state, event_id="old-schema")
    stale.additional_kwargs = {
        **stale.additional_kwargs,
        "goal_message_schema_version": GOAL_MESSAGE_SCHEMA_VERSION - 1,
    }
    messages: list[object] = [HumanMessage(content="continue"), stale]
    middleware = GoalToolsMiddleware()

    first = middleware._notice_update(
        cast("AgentState[Any]", {**state, "messages": messages})
    )

    assert first is not None
    fresh = first["messages"][0]
    info = goal_state_notice_info(fresh)
    assert info is not None
    assert info["schema_version"] == GOAL_MESSAGE_SCHEMA_VERSION
    assert "<goal_objective>ship it</goal_objective>" in fresh.content

    # Having appended the current notice, the next two boundaries must be no-ops.
    settled = [*messages, fresh]
    for _ in range(2):
        assert (
            middleware._notice_update(
                cast("AgentState[Any]", {**state, "messages": settled})
            )
            is None
        )


def test_notice_below_summarization_cutoff_is_rewritten() -> None:
    """A notice summarization has scrolled past is not authoritative.

    Summarization leaves `state["messages"]` intact and applies its cutoff only
    when building a request, so a matching notice below the cutoff is present in
    persisted history yet invisible to the model. `before_model` must still write a
    current one rather than leaving the transient re-pin to carry the objective on
    every later turn.
    """
    state: dict[str, object] = {
        "_goal_objective": "ship it",
        "_goal_status": "active",
        "_goal_rubric": "tests pass",
    }
    notice = build_goal_state_notice(state, event_id="in-view")
    messages: list[object] = [HumanMessage(content="continue"), notice]
    middleware = GoalToolsMiddleware()

    in_view = middleware._notice_update(
        cast("AgentState[Any]", {**state, "messages": messages})
    )
    trimmed_away = middleware._notice_update(
        cast(
            "AgentState[Any]",
            {
                **state,
                "messages": messages,
                "_summarization_event": {"cutoff_index": 2},
            },
        )
    )

    at_cutoff = middleware._notice_update(
        cast(
            "AgentState[Any]",
            {
                **state,
                "messages": messages,
                "_summarization_event": {"cutoff_index": 1},
            },
        )
    )

    assert in_view is None
    assert trimmed_away is not None
    assert "<goal_objective>ship it</goal_objective>" in (
        trimmed_away["messages"][0].content
    )
    # The comparison is `>=`, not `>`: `_effective_conversation` slices with
    # `messages[cutoff:]`, so the message at exactly `cutoff` is the first one
    # still visible and its notice remains authoritative. Weakening this to `>`
    # appends a redundant notice whenever summarization lands on the boundary.
    assert at_cutoff is None


@pytest.mark.parametrize("cutoff", [-1, 99])
def test_notice_update_repairs_malformed_summarization_cutoff(cutoff: int) -> None:
    """A restored invalid event is cleared alongside a fresh durable notice."""
    state: dict[str, object] = {
        "_goal_objective": "ship it",
        "_goal_status": "active",
    }
    notice = build_goal_state_notice(state, event_id="persisted")
    update = GoalToolsMiddleware._notice_update(
        cast(
            "AgentState[Any]",
            {
                **state,
                "messages": [notice, HumanMessage(content="continue")],
                "_summarization_event": {
                    "summary_message": HumanMessage(content="summary"),
                    "cutoff_index": cutoff,
                },
            },
        )
    )

    assert update is not None
    assert update["_summarization_event"] is None
    assert goal_state_notice_info(update["messages"][0]) is not None


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
    """Middleware should expose only the constrained write-side `update_goal` tool."""
    middleware = GoalToolsMiddleware()
    assert [tool.name for tool in middleware.tools] == ["update_goal"]
