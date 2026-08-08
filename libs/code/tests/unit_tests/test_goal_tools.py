"""Unit tests for goal tools middleware."""

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, NamedTuple, cast, get_type_hints

import pytest
from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command

from deepagents_code.goal_state_notice import (
    GOAL_MESSAGE_SCHEMA_VERSION,
    build_goal_continuation,
    build_goal_state_notice,
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


def test_stale_schema_notice_is_replaced_then_settles() -> None:
    """A resumed thread holding a prior-schema notice gets a current one, once.

    Prior notices could truncate required text or instruct the model to call read
    tools that no longer exist, so a resumed thread must not keep treating one as
    authoritative. The replacement must also converge: the notice channel is
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

    assert in_view is None
    assert trimmed_away is not None
    assert "<goal_objective>ship it</goal_objective>" in (
        trimmed_away["messages"][0].content
    )


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
