"""Tests for persisted goal-state notice reconciliation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents_code.app import DeepAgentsApp
from deepagents_code.goal_state_notice import (
    build_goal_state_notice,
    goal_state_notice_info,
)


def _active_state() -> dict[str, object]:
    return {
        "_goal_objective": "ship it",
        "_goal_status": "active",
        "_goal_rubric": "tests pass",
    }


def _serialized(message: HumanMessage) -> dict[str, object]:
    return {
        "type": "human",
        "content": message.content,
        "id": message.id,
        "additional_kwargs": dict(message.additional_kwargs),
    }


async def test_active_paused_active_persists_three_append_events() -> None:
    """A return to an earlier state does not reuse or replace its first event."""
    updater = SimpleNamespace(aupdate_state=AsyncMock())
    app = DeepAgentsApp(agent=MagicMock())
    app._agent = updater
    app._lc_thread_id = "thread-1"
    states = [
        {"_goal_objective": "ship it", "_goal_status": "active"},
        {"_goal_objective": "ship it", "_goal_status": "paused"},
        {"_goal_objective": "ship it", "_goal_status": "active"},
    ]

    for state in states:
        notice = build_goal_state_notice(state)
        assert await app._persist_goal_rubric_state(
            notice=notice,
            state_update=state,
        )

    assert updater.aupdate_state.await_count == 3
    notices = [
        awaited.args[1]["messages"][0]
        for awaited in updater.aupdate_state.await_args_list
    ]
    assert len({notice.id for notice in notices}) == 3
    assert (
        notices[0].additional_kwargs["state_fingerprint"]
        == notices[2].additional_kwargs["state_fingerprint"]
    )


async def test_legacy_active_thread_backfills_notice() -> None:
    """An active checkpoint without a notice is repaired before model use."""
    updater = SimpleNamespace(aupdate_state=AsyncMock())
    app = DeepAgentsApp(agent=MagicMock())
    app._agent = updater
    app._lc_thread_id = "thread-1"
    state = {**_active_state(), "messages": []}

    with patch.object(app, "_get_thread_state_values", AsyncMock(return_value=state)):
        assert await app._ensure_goal_state_notice()

    updater.aupdate_state.assert_awaited_once()
    update = updater.aupdate_state.await_args.args[1]
    assert set(update) == {"messages"}
    assert goal_state_notice_info(update["messages"][0]) is not None


async def test_matching_remote_notice_is_not_duplicated() -> None:
    """Serialized remote checkpoints use metadata for idempotent matching."""
    updater = SimpleNamespace(aupdate_state=AsyncMock())
    app = DeepAgentsApp(agent=MagicMock())
    app._agent = updater
    app._lc_thread_id = "thread-1"
    state = _active_state()
    notice = build_goal_state_notice(state, event_id="goal-event-1")
    checkpoint = {**state, "messages": [_serialized(notice)]}

    with patch.object(
        app,
        "_get_thread_state_values",
        AsyncMock(return_value=checkpoint),
    ):
        assert await app._ensure_goal_state_notice()

    updater.aupdate_state.assert_not_awaited()


async def test_stale_notice_appends_current_state() -> None:
    """A newer checkpoint state supersedes an older canonical notice."""
    updater = SimpleNamespace(aupdate_state=AsyncMock())
    app = DeepAgentsApp(agent=MagicMock())
    app._agent = updater
    app._lc_thread_id = "thread-1"
    stale = build_goal_state_notice(
        {"_goal_objective": "ship it", "_goal_status": "paused"},
        event_id="goal-event-paused",
    )
    checkpoint = {**_active_state(), "messages": [stale]}

    with patch.object(
        app,
        "_get_thread_state_values",
        AsyncMock(return_value=checkpoint),
    ):
        assert await app._ensure_goal_state_notice()

    current = updater.aupdate_state.await_args.args[1]["messages"][0]
    assert "Goal status: active" in current.content
    assert current.id != stale.id


async def test_compaction_cutoff_repins_once() -> None:
    """A matching notice before the active cutoff is appended once after it."""
    updater = SimpleNamespace(aupdate_state=AsyncMock())
    app = DeepAgentsApp(agent=MagicMock())
    app._agent = updater
    app._lc_thread_id = "thread-1"
    state = _active_state()
    old_notice = build_goal_state_notice(state, event_id="goal-event-old")
    user = HumanMessage(content="continue", id="user-1")
    event = {
        "summary_message": HumanMessage(
            content="summary",
            additional_kwargs={"lc_source": "summarization"},
        ),
        "cutoff_index": 1,
    }
    checkpoint = {
        **state,
        "messages": [old_notice, user],
        "_summarization_event": event,
    }

    fetch = AsyncMock(return_value=checkpoint)
    with patch.object(app, "_get_thread_state_values", fetch):
        assert await app._ensure_goal_state_notice()
    repinned = updater.aupdate_state.await_args.args[1]["messages"][0]
    assert repinned.id != old_notice.id

    updater.aupdate_state.reset_mock()
    checkpoint["messages"] = [old_notice, user, repinned]
    with patch.object(
        app,
        "_get_thread_state_values",
        AsyncMock(return_value=checkpoint),
    ):
        assert await app._ensure_goal_state_notice()
    updater.aupdate_state.assert_not_awaited()


@pytest.mark.parametrize("parallel_calls", [False, True])
async def test_notice_waits_for_complete_tool_result_batch(
    parallel_calls: bool,
) -> None:
    """No human notice is inserted inside a single or parallel tool batch."""
    updater = SimpleNamespace(aupdate_state=AsyncMock())
    app = DeepAgentsApp(agent=MagicMock())
    app._agent = updater
    app._lc_thread_id = "thread-1"
    tool_calls = [{"name": "one", "args": {}, "id": "call-1"}]
    if parallel_calls:
        tool_calls.append({"name": "two", "args": {}, "id": "call-2"})
    assistant = AIMessage(content="", tool_calls=tool_calls)
    partial = [assistant, ToolMessage(content="done", tool_call_id="call-1")]
    if not parallel_calls:
        partial = [assistant]
    checkpoint = {**_active_state(), "messages": partial}

    with patch.object(
        app,
        "_get_thread_state_values",
        AsyncMock(return_value=checkpoint),
    ):
        assert not await app._ensure_goal_state_notice()
    updater.aupdate_state.assert_not_awaited()

    complete = [assistant, ToolMessage(content="done", tool_call_id="call-1")]
    if parallel_calls:
        complete.append(ToolMessage(content="done", tool_call_id="call-2"))
    checkpoint["messages"] = complete
    with patch.object(
        app,
        "_get_thread_state_values",
        AsyncMock(return_value=checkpoint),
    ):
        assert await app._ensure_goal_state_notice()
    updater.aupdate_state.assert_awaited_once()


async def test_remote_state_and_notice_share_one_update() -> None:
    """Remote TUI transitions use one attributed state-plus-message write."""
    from deepagents_code.client.remote_client import RemoteAgent

    remote = MagicMock(spec=RemoteAgent)
    remote.aensure_thread = AsyncMock()
    remote.aupdate_state = AsyncMock()
    app = DeepAgentsApp(agent=remote)
    app._lc_thread_id = "thread-1"
    state = _active_state()
    notice = build_goal_state_notice(state, event_id="goal-event-remote")

    assert await app._persist_goal_rubric_state(
        notice=notice,
        state_update=dict(state),
    )

    remote.aensure_thread.assert_awaited_once_with(
        {"configurable": {"thread_id": "thread-1"}}
    )
    remote.aupdate_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "thread-1"}},
        {**state, "messages": [notice]},
        as_node="model",
    )
