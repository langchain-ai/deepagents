"""Tests for RemoteAgent, _convert_message_data, and helpers."""

import asyncio
import itertools
import logging
import uuid
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from deepagents_code._env_vars import LANGSMITH_REPLICA_PROJECTS
from deepagents_code.client.remote_client import (
    RemoteAgent,
    _cancelled_tool_messages,
    _convert_interrupts,
    _convert_message_data,
    _prepare_config,
    agent_error_type,
    format_agent_exception,
)

_TEST_THREAD_ID = "01966f3a-0000-7000-8000-000000000001"

_COMPACTED_RESULT = {
    "status": "compacted",
    "messages_offloaded": 2,
    "messages_kept": 3,
    "tokens_before": 100,
    "tokens_after": 40,
    "archive_path": "/conversation_history/thread.md",
    "archive_ephemeral": False,
    "error": None,
}
"""A well-formed `compacted` result, for tests that perturb one field."""


# ---------------------------------------------------------------------------
# _prepare_config
# ---------------------------------------------------------------------------


class TestPrepareConfig:
    def test_preserves_top_level_tags(self) -> None:
        """Every dcode config crosses this seam before reaching the server.

        Trace tags such as `dcode:resume` live at the top level, so narrowing
        this to a whitelist of known keys would drop them and make trace
        grouping a silent no-op with its own tests still green.
        """
        result = _prepare_config(
            {"configurable": {"thread_id": "t1"}, "tags": ["dcode:resume"]}
        )
        assert result["tags"] == ["dcode:resume"]


# ---------------------------------------------------------------------------
# _convert_message_data
# ---------------------------------------------------------------------------


class TestConvertMessageData:
    def test_ai_message_with_tool_call_chunks(self) -> None:
        msg = _convert_message_data(
            {
                "type": "AIMessageChunk",
                "content": "",
                "id": "m1",
                "tool_call_chunks": [
                    {"name": "search", "args": '{"q":', "id": "tc1", "index": 0}
                ],
            }
        )
        assert isinstance(msg, AIMessageChunk)
        tc_blocks = [
            b for b in msg.content_blocks if b.get("type") == "tool_call_chunk"
        ]
        assert len(tc_blocks) == 1
        assert tc_blocks[0]["name"] == "search"
        assert tc_blocks[0]["args"] == '{"q":'

    def test_ai_message_with_string_args_tool_calls(self) -> None:
        msg = _convert_message_data(
            {
                "type": "ai",
                "content": "",
                "id": "m1",
                "tool_calls": [{"name": "ls", "args": '{"path":"/"', "id": "tc1"}],
            }
        )
        assert isinstance(msg, AIMessageChunk)
        tc_blocks = [
            b for b in msg.content_blocks if b.get("type") == "tool_call_chunk"
        ]
        assert len(tc_blocks) == 1

    def test_tool_message_forwards_additional_kwargs(self) -> None:
        """Markers set server-side survive the conversion.

        The TUI always runs against a server, so a marker dropped here never
        reaches the adapter. `AUTO_DENIED_METADATA_KEY` is the live consumer.
        """
        from deepagents_code.auto_mode import AUTO_DENIED_METADATA_KEY

        msg = _convert_message_data(
            {
                "type": "tool",
                "content": "Auto denied [credential_access]: not authorized",
                "tool_call_id": "tc1",
                "name": "onepassword_authenticate",
                "status": "error",
                "id": "m4",
                "additional_kwargs": {AUTO_DENIED_METADATA_KEY: True},
            }
        )
        assert isinstance(msg, ToolMessage)
        assert msg.additional_kwargs[AUTO_DENIED_METADATA_KEY] is True

    def test_tool_message_additional_kwargs_defaults_to_empty(self) -> None:
        """A missing or non-dict `additional_kwargs` does not discard the message."""
        msg = _convert_message_data({"type": "tool", "id": "m1"})
        assert isinstance(msg, ToolMessage)
        assert msg.additional_kwargs == {}

        msg = _convert_message_data(
            {"type": "tool", "id": "m1", "additional_kwargs": "not-a-dict"}
        )
        assert isinstance(msg, ToolMessage)
        assert msg.additional_kwargs == {}


# ---------------------------------------------------------------------------
# _convert_interrupts
# ---------------------------------------------------------------------------


class TestConvertInterrupts:
    def test_dicts_to_interrupt_objects(self) -> None:
        from langgraph.types import Interrupt

        result = _convert_interrupts([{"value": {"type": "ask_user"}, "id": "int-1"}])
        assert len(result) == 1
        assert isinstance(result[0], Interrupt)
        assert result[0].value == {"type": "ask_user"}
        assert result[0].id == "int-1"

    def test_interrupt_objects_passed_through(self) -> None:
        from langgraph.types import Interrupt

        obj = Interrupt(value="test", id="int-2")
        result = _convert_interrupts([obj])
        assert result[0] is obj

    def test_dict_without_value_passed_through(self) -> None:
        raw = [{"id": "x", "other": 123}]
        result = _convert_interrupts(raw)
        assert result[0] == {"id": "x", "other": 123}

    def test_interrupt_dict_missing_id_defaults_to_empty(self) -> None:
        from langgraph.types import Interrupt

        result = _convert_interrupts([{"value": "confirm"}])
        assert isinstance(result[0], Interrupt)
        assert result[0].value == "confirm"
        assert result[0].id == ""


# ---------------------------------------------------------------------------
# Helpers for RemoteAgent tests
# ---------------------------------------------------------------------------


def _make_agent(
    events: Sequence[tuple[tuple[str, ...], str, Any]],
) -> RemoteAgent:
    """Create a RemoteAgent with a mock RemoteGraph yielding events."""
    agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
    mock_graph = MagicMock()

    async def fake_astream(  # noqa: RUF029
        input: Any,  # noqa: A002, ANN401, ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> Any:  # noqa: ANN401
        for ev in events:
            yield ev

    mock_graph.astream = fake_astream
    agent._graph = mock_graph
    agent._workspaces[_TEST_THREAD_ID] = {"workspace_id": "test-workspace"}
    return agent


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": _TEST_THREAD_ID}}


def _make_capturing_agent() -> tuple[RemoteAgent, dict[str, Any]]:
    """RemoteAgent whose mock graph records the kwargs passed to `astream`."""
    agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
    captured: dict[str, Any] = {}
    mock_graph = MagicMock()

    async def fake_astream(  # noqa: RUF029
        input: Any,  # noqa: A002, ANN401, ARG001
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        captured.update(kwargs)
        for ev in ():  # async generator that yields nothing
            yield ev

    mock_graph.astream = fake_astream
    agent._graph = mock_graph
    agent._workspaces[_TEST_THREAD_ID] = {"workspace_id": "test-workspace"}
    return agent, captured


class TestRemoteAgentReplicaForwarding:
    """`astream` forwards the LangSmith replica project to the server SDK.

    The server mirrors a run to an extra project only via the SDK's
    `langsmith_tracing` field, so these lock the exact kwarg name and payload
    shape `RemoteGraph.astream` (and thus `client.runs.stream`) expects.

    `test_forwards_replica_project` / `test_no_kwarg_when_unset` assert the
    payload against a mock graph that swallows any kwarg, so they verify only
    the `RemoteAgent` side of the contract. `test_sdk_accepts_langsmith_tracing`
    pins the *other* side — that the real SDK still accepts the kwarg and shape —
    so a future SDK rename surfaces here rather than silently dropping replicas.
    """

    async def test_forwards_replica_project(self, monkeypatch) -> None:
        """A configured replica is passed through as `langsmith_tracing`."""
        monkeypatch.setenv(LANGSMITH_REPLICA_PROJECTS, "mason-dual-trace")
        agent, captured = _make_capturing_agent()
        async for _ in agent.astream({"messages": []}, config=_config()):
            pass
        assert captured["langsmith_tracing"] == {"project_name": "mason-dual-trace"}


# ---------------------------------------------------------------------------
# RemoteAgent — astream delegation
# ---------------------------------------------------------------------------


class TestRemoteAgentAstream:
    async def test_updates_with_interrupt_converted(self) -> None:
        """Interrupt dicts in updates events are converted to Interrupt."""
        from langgraph.types import Interrupt

        events = [
            (
                (),
                "updates",
                {"__interrupt__": [{"value": {"type": "ask_user"}, "id": "int-1"}]},
            )
        ]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert len(results) == 1
        interrupts = results[0][2]["__interrupt__"]
        assert isinstance(interrupts[0], Interrupt)

    async def test_updates_without_interrupt_passed_through(self) -> None:
        """Regular updates events pass through unchanged."""
        events = [((), "updates", {"agent": {"messages": []}})]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert len(results) == 1
        assert results[0][1] == "updates"
        assert results[0][2] == {"agent": {"messages": []}}


# ---------------------------------------------------------------------------
# RemoteAgent — aget_state
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RemoteAgent — aupdate_state
# ---------------------------------------------------------------------------


class TestRemoteAgentCancelActiveRuns:
    """`acancel_active_runs` exposes best-effort remote run cancellation."""

    async def test_cancels_running_and_pending_runs(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        runs_list = AsyncMock(
            side_effect=[
                [{"run_id": "run-1"}],
                [{"run_id": "run-2"}],
            ]
        )
        runs_cancel = AsyncMock()
        mock_runs = MagicMock()
        mock_runs.list = runs_list
        mock_runs.cancel = runs_cancel
        mock_client = MagicMock()
        mock_client.runs = mock_runs
        mock_graph = MagicMock()
        mock_graph._validate_client.return_value = mock_client
        agent._graph = mock_graph

        await agent.acancel_active_runs(_config())

        assert runs_list.await_count == 2
        assert runs_cancel.await_count == 2
        assert {call.args[1] for call in runs_cancel.await_args_list} == {
            "run-1",
            "run-2",
        }

    async def test_raises_when_thread_id_missing(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        with pytest.raises(ValueError, match="thread_id"):
            await agent.acancel_active_runs({"configurable": {}})


def _conflict_error() -> Exception:
    """Build a `ConflictError` (HTTP 409) for tests."""
    import httpx
    from langgraph_sdk.errors import ConflictError

    request = httpx.Request("POST", "http://localhost:8123/threads/x/state")
    response = httpx.Response(409, request=request)
    return ConflictError("Thread busy", response=response, body=None)


class TestRemoteAgentUpdateStateConflictRecovery:
    """`aupdate_state` cancels in-flight runs on 409 and retries once."""

    def _agent_with_client(
        self,
        *,
        runs_list: AsyncMock,
        runs_cancel: AsyncMock,
        update_side_effect: list[Any],
    ) -> tuple[RemoteAgent, MagicMock]:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        mock_graph.aupdate_state = AsyncMock(side_effect=update_side_effect)
        mock_runs = MagicMock()
        mock_runs.list = runs_list
        mock_runs.cancel = runs_cancel
        mock_client = MagicMock()
        mock_client.runs = mock_runs
        mock_graph._validate_client.return_value = mock_client
        agent._graph = mock_graph
        return agent, mock_graph

    async def test_cancels_all_active_runs_then_retries(self) -> None:
        runs_list = AsyncMock(
            side_effect=[
                [{"run_id": "run-1"}, {"run_id": "run-2"}],  # running
                [{"run_id": "run-3"}],  # pending
            ]
        )
        runs_cancel = AsyncMock()
        agent, mock_graph = self._agent_with_client(
            runs_list=runs_list,
            runs_cancel=runs_cancel,
            update_side_effect=[_conflict_error(), None],
        )

        await agent.aupdate_state(_config(), {"messages": []})

        assert runs_list.await_count == 2
        assert runs_cancel.await_count == 3
        cancelled_ids = {call.args[1] for call in runs_cancel.await_args_list}
        assert cancelled_ids == {"run-1", "run-2", "run-3"}
        # wait=True + action="interrupt" are contractual — `wait` is what
        # actually settles the thread before the retry.
        for call in runs_cancel.await_args_list:
            assert call.kwargs == {"wait": True, "action": "interrupt"}
        assert mock_graph.aupdate_state.await_count == 2

    async def test_no_active_runs_still_retries(self) -> None:
        runs_list = AsyncMock(return_value=[])
        runs_cancel = AsyncMock()
        agent, mock_graph = self._agent_with_client(
            runs_list=runs_list,
            runs_cancel=runs_cancel,
            update_side_effect=[_conflict_error(), None],
        )

        await agent.aupdate_state(_config(), {"messages": []})

        assert runs_cancel.await_count == 0
        assert mock_graph.aupdate_state.await_count == 2

    async def test_retry_still_conflict_raises(self) -> None:
        runs_list = AsyncMock(return_value=[])
        runs_cancel = AsyncMock()
        agent, mock_graph = self._agent_with_client(
            runs_list=runs_list,
            runs_cancel=runs_cancel,
            update_side_effect=[_conflict_error(), _conflict_error()],
        )

        from langgraph_sdk.errors import ConflictError

        with pytest.raises(ConflictError):
            await agent.aupdate_state(_config(), {"messages": []})
        assert mock_graph.aupdate_state.await_count == 2

    async def test_cancel_timeout_still_retries(self) -> None:
        import asyncio

        async def slow_cancel(*_args: Any, **_kwargs: Any) -> None:
            await asyncio.sleep(60)  # exceeds wait_for timeout

        runs_list = AsyncMock(return_value=[{"run_id": "run-1"}])
        runs_cancel = AsyncMock(side_effect=slow_cancel)
        agent, mock_graph = self._agent_with_client(
            runs_list=runs_list,
            runs_cancel=runs_cancel,
            update_side_effect=[_conflict_error(), None],
        )

        with patch(
            "deepagents_code.client.remote_client._RUN_CANCEL_WAIT_SECONDS", 0.01
        ):
            await agent.aupdate_state(_config(), {"messages": []})

        assert mock_graph.aupdate_state.await_count == 2

    async def test_cancel_non_timeout_exception_is_swallowed(self) -> None:
        runs_list = AsyncMock(side_effect=[[{"run_id": "run-1"}], []])
        runs_cancel = AsyncMock(side_effect=RuntimeError("server hiccup"))
        agent, mock_graph = self._agent_with_client(
            runs_list=runs_list,
            runs_cancel=runs_cancel,
            update_side_effect=[_conflict_error(), None],
        )

        await agent.aupdate_state(_config(), {"messages": []})

        assert runs_cancel.await_count == 1
        assert mock_graph.aupdate_state.await_count == 2

    async def test_runs_list_partial_failure_still_retries(self) -> None:
        # First status list raises; second returns runs. Recovery should still
        # cancel what it can find and retry.
        runs_list = AsyncMock(side_effect=[RuntimeError("boom"), [{"run_id": "run-2"}]])
        runs_cancel = AsyncMock()
        agent, mock_graph = self._agent_with_client(
            runs_list=runs_list,
            runs_cancel=runs_cancel,
            update_side_effect=[_conflict_error(), None],
        )

        await agent.aupdate_state(_config(), {"messages": []})

        assert runs_list.await_count == 2
        assert runs_cancel.await_count == 1
        assert runs_cancel.await_args_list[0].args[1] == "run-2"
        assert mock_graph.aupdate_state.await_count == 2

    async def test_runs_list_total_failure_skips_cancel(self) -> None:
        # Both status calls raise. With nothing listed, no cancels happen and
        # the retry surfaces the persistent conflict.
        runs_list = AsyncMock(side_effect=[RuntimeError("boom"), RuntimeError("boom")])
        runs_cancel = AsyncMock()
        agent, mock_graph = self._agent_with_client(
            runs_list=runs_list,
            runs_cancel=runs_cancel,
            update_side_effect=[_conflict_error(), _conflict_error()],
        )

        from langgraph_sdk.errors import ConflictError

        with pytest.raises(ConflictError):
            await agent.aupdate_state(_config(), {"messages": []})
        runs_cancel.assert_not_called()
        assert mock_graph.aupdate_state.await_count == 2

    async def test_validate_client_raises_skips_cancel_and_retries(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        mock_graph.aupdate_state = AsyncMock(
            side_effect=[_conflict_error(), _conflict_error()]
        )
        mock_graph._validate_client.side_effect = RuntimeError("no client")
        agent._graph = mock_graph

        from langgraph_sdk.errors import ConflictError

        with pytest.raises(ConflictError):
            await agent.aupdate_state(_config(), {"messages": []})
        assert mock_graph.aupdate_state.await_count == 2

    async def test_runs_without_run_id_are_skipped(self) -> None:
        runs_list = AsyncMock(
            side_effect=[
                # Mixed shapes: missing key, None id, non-dict — all skipped.
                [{"run_id": "ok"}, {"run_id": None}, {"status": "running"}, "garbage"],
                [],
            ]
        )
        runs_cancel = AsyncMock()
        agent, mock_graph = self._agent_with_client(
            runs_list=runs_list,
            runs_cancel=runs_cancel,
            update_side_effect=[_conflict_error(), None],
        )

        await agent.aupdate_state(_config(), {"messages": []})

        assert runs_cancel.await_count == 1
        assert runs_cancel.await_args_list[0].args[1] == "ok"
        assert mock_graph.aupdate_state.await_count == 2


class TestCancelledToolMessages:
    def test_supports_serialized_messages_and_ignores_answered_calls(self) -> None:
        values = {
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {"name": "shell", "args": {}, "id": "answered"},
                        {"name": "shell", "args": {}, "id": "pending"},
                    ],
                },
                {
                    "type": "tool",
                    "content": "done",
                    "tool_call_id": "answered",
                },
            ]
        }

        cancelled = _cancelled_tool_messages(values)

        assert [message.tool_call_id for message in cancelled] == ["pending"]
        assert cancelled[0].status == "error"

    @pytest.mark.parametrize("values", [None, [], {}, {"messages": "invalid"}])
    def test_ignores_state_without_a_message_list(self, values: object) -> None:
        assert _cancelled_tool_messages(values) == []

    def test_leaves_earlier_interrupted_turns_dangling(self) -> None:
        """Only the trailing turn is answered.

        Interrupt recovery persists a partial `AIMessage` carrying its
        in-flight `tool_calls` and then closes the turn with a cancellation
        notice, so history routinely holds calls that are unanswered by
        design. Answering one here would append its `tool_result` after
        unrelated messages, which the provider rejects.
        """
        values = {
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"name": "shell", "args": {}, "id": "interrupted"}],
                },
                {"type": "human", "content": "Task interrupted by user."},
                {"type": "human", "content": "try again"},
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"name": "shell", "args": {}, "id": "pending"}],
                },
            ]
        }

        cancelled = _cancelled_tool_messages(values)

        assert [message.tool_call_id for message in cancelled] == ["pending"]

    def test_ignores_a_turn_that_already_closed(self) -> None:
        values = {
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"name": "shell", "args": {}, "id": "interrupted"}],
                },
                {"type": "human", "content": "Task interrupted by user."},
            ]
        }

        assert _cancelled_tool_messages(values) == []


def _pending_state(values: dict[str, Any]) -> SimpleNamespace:
    """Snapshot stub for a thread with a queued `tools` step."""
    return SimpleNamespace(
        values=values, next=("tools",), tasks=(object(),), interrupts=()
    )


def _idle_state() -> SimpleNamespace:
    """Snapshot stub for a thread with nothing left to run."""
    return SimpleNamespace(values={}, next=(), tasks=(), interrupts=())


class TestRemoteAgentAbandonPendingWork:
    def _agent_with_states(self, *states: Any) -> tuple[RemoteAgent, MagicMock]:
        """RemoteAgent whose mock graph returns `states` from successive reads."""
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_client = MagicMock()
        mock_client.runs.list = AsyncMock(return_value=[])
        mock_graph = MagicMock()
        mock_graph._validate_client.return_value = mock_client
        mock_graph.aupdate_state = AsyncMock()
        mock_graph.aget_state = AsyncMock(side_effect=list(states))
        agent._graph = mock_graph
        return agent, mock_graph

    async def test_cancels_runs_clears_checkpoint_and_verifies(self) -> None:
        agent, mock_graph = self._agent_with_states(
            _pending_state({"messages": []}), _idle_state()
        )

        await agent.aabandon_pending_work(_config())

        assert mock_graph._validate_client.return_value.runs.list.await_count == 2
        mock_graph.aupdate_state.assert_awaited_once()
        state_update = mock_graph.aupdate_state.await_args
        assert state_update is not None
        assert state_update.args[1] is None
        assert state_update.kwargs == {"as_node": "__end__"}
        assert mock_graph.aget_state.await_count == 2

    async def test_terminalizes_dangling_tool_call_before_clearing(self) -> None:
        from langchain_core.messages import AIMessage

        agent, mock_graph = self._agent_with_states(
            _pending_state(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{"name": "shell", "args": {}, "id": "call-1"}],
                        )
                    ]
                }
            ),
            _idle_state(),
        )

        await agent.aabandon_pending_work(_config())

        assert mock_graph.aupdate_state.await_count == 2
        message_update = mock_graph.aupdate_state.await_args_list[0]
        assert message_update.kwargs == {"as_node": "tools"}
        cancelled = message_update.args[1]["messages"][0]
        assert cancelled.tool_call_id == "call-1"
        assert cancelled.status == "error"
        assert mock_graph.aupdate_state.await_args_list[1].args[1] is None

    async def test_raises_when_pending_work_remains(self) -> None:
        agent, _ = self._agent_with_states(
            _pending_state({"messages": []}), _pending_state({"messages": []})
        )

        with pytest.raises(RuntimeError, match="Pending graph work remained"):
            await agent.aabandon_pending_work(_config())


class TestRemoteAgentStore:
    async def test_aput_store_item_uses_unindexed_put(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        store = SimpleNamespace(put_item=AsyncMock())
        client = SimpleNamespace(store=store)
        graph = MagicMock()
        graph._validate_client.return_value = client
        agent._graph = graph

        await agent.aput_store_item(("ns",), "key", {"auto_approve": True})

        store.put_item.assert_awaited_once_with(
            ("ns",),
            "key",
            {"auto_approve": True},
            index=False,
        )

    async def test_aput_store_item_logs_and_reraises(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        store = SimpleNamespace(put_item=AsyncMock(side_effect=RuntimeError("boom")))
        client = SimpleNamespace(store=store)
        graph = MagicMock()
        graph._validate_client.return_value = client
        agent._graph = graph

        with (
            caplog.at_level("DEBUG", logger="deepagents_code.client.remote_client"),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await agent.aput_store_item(("ns",), "key", {"auto_approve": True})

        assert "Failed to write store item ns/key" in caplog.text


class TestRemoteAgentEnsureThread:
    """Verify remote thread registration before state writes."""

    async def test_creates_thread_with_do_nothing(self) -> None:
        """Creates the remote thread idempotently before cold-resume updates."""
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_threads = MagicMock()
        mock_threads.create = AsyncMock()
        mock_client = MagicMock()
        mock_client.threads = mock_threads
        mock_graph = MagicMock()
        mock_graph._validate_client.return_value = mock_client
        agent._graph = mock_graph

        await agent.aensure_thread(
            {
                "configurable": {"thread_id": _TEST_THREAD_ID},
                "metadata": {"assistant_id": "agent"},
            }
        )

        kwargs = mock_threads.create.call_args.kwargs
        uuid.UUID(kwargs["thread_id"])
        assert kwargs["if_exists"] == "do_nothing"
        assert kwargs["metadata"] == {"assistant_id": "agent"}
        assert kwargs["graph_id"] == "agent"


# ---------------------------------------------------------------------------
# RemoteAgent — with_config
# ---------------------------------------------------------------------------


class TestFormatAgentException:
    """Cover the rendering helper for agent-stream exceptions."""

    def test_remote_exception_dict_payload(self) -> None:
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException(
            {"error": "ToolException", "message": "An internal error occurred"}
        )
        assert (
            format_agent_exception(exc) == "ToolException: An internal error occurred"
        )

    def test_remote_exception_dict_payload_no_message(self) -> None:
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException({"error": "ToolException"})
        assert format_agent_exception(exc) == "ToolException"

    def test_remote_exception_dict_payload_empty_message(self) -> None:
        """Falsy `message` still falls through to the error-type-only branch."""
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException({"error": "ToolException", "message": ""})
        assert format_agent_exception(exc) == "ToolException"

    def test_remote_exception_dict_payload_non_string_error(self) -> None:
        """Non-string `error` keys must not crash; class name stands in."""
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException({"error": 500, "message": "boom"})
        # `agent_error_type` ignores the non-string `error` and uses the class
        # name, so the message still renders cleanly.
        assert format_agent_exception(exc) == "RemoteException: boom"

    def test_remote_exception_dict_payload_empty_dict(self) -> None:
        """Empty payload dict resolves `error` to the exception class name."""
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException({})
        # `payload.get("error") or type(exc).__name__` → "RemoteException",
        # and `message` is None so the err-only branch returns the class.
        assert format_agent_exception(exc) == "RemoteException"

    def test_remote_exception_non_dict_payload(self) -> None:
        """`RemoteException("string")` is not the dict shape; uses `str(exc)`."""
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException("just a string")
        assert format_agent_exception(exc) == "just a string"

    def test_plain_exception_uses_str(self) -> None:
        assert format_agent_exception(ValueError("bad thing")) == "bad thing"

    def test_exception_without_message_falls_back_to_type(self) -> None:
        class _BoomError(Exception):
            pass

        assert format_agent_exception(_BoomError()) == "_BoomError"


class TestAgentErrorType:
    """Cover the shared error-type extraction used for UI dispatch."""

    def test_dict_payload_error_key_wins(self) -> None:
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException({"error": "PermissionDeniedError", "message": "x"})
        assert agent_error_type(exc) == "PermissionDeniedError"


def _offload_graph(http: SimpleNamespace) -> SimpleNamespace:
    """Build a graph stub that also satisfies `aensure_thread`.

    `aoffload` registers the thread before its first POST, so a stub that only
    carries `client.http` no longer suffices. `threads.create` is recorded so
    tests can assert the registration happened, and happened first.
    """
    threads = SimpleNamespace(create=AsyncMock(return_value=None))
    client = SimpleNamespace(http=http, threads=threads)
    return SimpleNamespace(
        client=client,
        _validate_client=lambda: client,
    )


class TestRemoteAgentWorkspace:
    """Workspace bindings stay thread-scoped and require explicit launch state."""

    async def test_bindings_are_cached_per_thread(self) -> None:
        agent = RemoteAgent("http://localhost:1234")
        post = AsyncMock(
            side_effect=[
                {"workspace": {"workspace_id": "first"}},
                {"workspace": {"workspace_id": "second"}},
            ]
        )
        graph = SimpleNamespace(client=SimpleNamespace(http=SimpleNamespace(post=post)))
        agent.set_workspace(
            "/workspace/project",
            {"enable_shell": True},
            config_fingerprint="config-fingerprint",
        )

        with patch.object(agent, "_get_graph", return_value=graph):
            first = await agent._workspace_for_thread(
                {"configurable": {"thread_id": "thread-1"}}
            )
            repeated = await agent._workspace_for_thread(
                {"configurable": {"thread_id": "thread-1"}}
            )
            second = await agent._workspace_for_thread(
                {"configurable": {"thread_id": "thread-2"}}
            )

        assert first == repeated == {"workspace_id": "first"}
        assert second == {"workspace_id": "second"}
        assert post.await_count == 2
        assert post.await_args_list[0].kwargs["json"] == {
            "cwd": "/workspace/project",
            "workspace_config": {"enable_shell": True},
            "config_fingerprint": "config-fingerprint",
        }

    async def test_binding_can_defer_policy_to_external_server(self) -> None:
        agent = RemoteAgent("http://localhost:1234")
        post = AsyncMock(return_value={"workspace": {"workspace_id": "remote"}})
        graph = SimpleNamespace(client=SimpleNamespace(http=SimpleNamespace(post=post)))
        agent.set_workspace("/workspace/project")

        with patch.object(agent, "_get_graph", return_value=graph):
            workspace = await agent._workspace_for_thread(
                {"configurable": {"thread_id": "thread-1"}}
            )

        assert workspace == {"workspace_id": "remote"}
        post.assert_awaited_once_with(
            "/dcode/threads/thread-1/workspace",
            json={"cwd": "/workspace/project"},
        )

    async def test_switch_preserves_destination_binding_and_returns_metadata(
        self,
    ) -> None:
        agent = RemoteAgent("http://localhost:1234")
        agent.set_workspace("/workspace/old")
        agent._workspaces["old"] = {"workspace_id": "old"}
        post = AsyncMock(
            return_value={
                "workspace": {"workspace_id": "new"},
                "mcp_server_info": [
                    {
                        "name": "docs",
                        "transport": "http",
                        "tools": [{"name": "search", "description": "Search docs"}],
                        "status": "ok",
                        "error": None,
                        "pending_reconnect": False,
                        "uses_oauth": False,
                    }
                ],
            }
        )
        graph = SimpleNamespace(client=SimpleNamespace(http=SimpleNamespace(post=post)))

        with patch.object(agent, "_get_graph", return_value=graph):
            info = await agent.aswitch_workspace(
                {"configurable": {"thread_id": "destination"}}, "/workspace/new"
            )
            binding = await agent._workspace_for_thread(
                {"configurable": {"thread_id": "destination"}}
            )

        assert agent._workspace_cwd == "/workspace/new"
        assert binding == {"workspace_id": "new"}
        assert post.await_count == 1
        assert info is not None
        assert info[0].name == "docs"
        assert info[0].tools[0].name == "search"

    async def test_switch_failure_preserves_workspace_state(self) -> None:
        agent = RemoteAgent("http://localhost:1234")
        agent.set_workspace("/workspace/old")
        agent._workspaces["old"] = {"workspace_id": "old"}
        graph = SimpleNamespace(
            client=SimpleNamespace(
                http=SimpleNamespace(post=AsyncMock(side_effect=RuntimeError("no")))
            )
        )

        with (
            patch.object(agent, "_get_graph", return_value=graph),
            pytest.raises(RuntimeError, match="no"),
        ):
            await agent.aswitch_workspace(
                {"configurable": {"thread_id": "destination"}}, "/workspace/new"
            )

        assert agent._workspace_cwd == "/workspace/old"
        assert agent._workspaces == {"old": {"workspace_id": "old"}}

    def test_policy_and_fingerprint_must_be_configured_together(self) -> None:
        agent = RemoteAgent("http://localhost:1234")

        with pytest.raises(ValueError, match="configured together"):
            agent.set_workspace("/workspace/project", {"enable_shell": True})

    async def test_binding_requires_explicit_workspace(self) -> None:
        agent = RemoteAgent("http://localhost:1234")

        with pytest.raises(RuntimeError, match="not configured"):
            await agent._workspace_for_thread(
                {"configurable": {"thread_id": "thread-1"}}
            )


class TestServerOffload:
    """The remote client transports operation data without graph state."""

    @pytest.fixture(autouse=True)
    def _bound_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def workspace_for_thread(  # noqa: RUF029
            _agent: RemoteAgent,
            _config: dict[str, Any],
        ) -> dict[str, Any]:
            return {"workspace_id": "test-workspace"}

        monkeypatch.setattr(RemoteAgent, "_workspace_for_thread", workspace_for_thread)

    async def test_cancellation_waits_for_server_acknowledgement(self) -> None:
        """Esc must not release the caller while server offload is still live."""
        request_started = asyncio.Event()
        cancel_started = asyncio.Event()
        acknowledge_cancel = asyncio.Event()

        async def post(path: str, **_kwargs: object) -> dict[str, object]:
            if path.endswith("/cancel"):
                cancel_started.set()
                await acknowledge_cancel.wait()
                return {"status": "cancelled"}
            request_started.set()
            await asyncio.Event().wait()
            return {}

        http = SimpleNamespace(post=AsyncMock(side_effect=post))
        graph = _offload_graph(http)
        agent = RemoteAgent("http://localhost:1234")

        with patch.object(agent, "_get_graph", return_value=graph):
            task = asyncio.create_task(
                agent.aoffload(
                    config={"configurable": {"thread_id": "thread"}},
                    context={},
                    fulfill_hook=AsyncMock(),
                )
            )
            await asyncio.wait_for(request_started.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(cancel_started.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            acknowledge_cancel.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert http.post.await_count == 2
        request_call, cancel_call = http.post.await_args_list
        operation_id = request_call.kwargs["json"]["operation_id"]
        assert cancel_call.args[0] == (
            f"/dcode/threads/thread/offload/{operation_id}/cancel"
        )

    async def test_registers_the_thread_before_the_first_request(self) -> None:
        """The operation must not be requested against an unregistered thread.

        Checkpoint persistence and HTTP thread registration are separate on the
        dev server, so a resumed thread has state on disk and no live row, and
        every request below would 404. Ordering is the whole point -- registering
        after the POST would not help -- so assert the call sequence rather than
        just that both calls happened.
        """
        calls: list[str] = []

        async def record_post(  # noqa: RUF029 -- must satisfy the async post signature
            *_args: object, **_kwargs: object
        ) -> dict[str, object]:
            calls.append("post")
            return {"status": "complete", "result": dict(_COMPACTED_RESULT)}

        async def record_create(  # noqa: RUF029 -- must satisfy the async create signature
            *_args: object, **_kwargs: object
        ) -> None:
            calls.append("create")

        http = SimpleNamespace(post=AsyncMock(side_effect=record_post))
        graph = _offload_graph(http)
        graph.client.threads.create.side_effect = record_create

        agent = RemoteAgent("http://localhost:1234")
        with patch.object(agent, "_get_graph", return_value=graph):
            await agent.aoffload(
                config={"configurable": {"thread_id": "thread"}},
                context={"model": "test:model"},
                fulfill_hook=AsyncMock(),
            )

        assert calls == ["create", "post"]
        create_kwargs = graph.client.threads.create.await_args.kwargs
        assert create_kwargs["thread_id"] == "thread"
        assert create_kwargs["if_exists"] == "do_nothing"

    async def test_fulfills_hook_and_returns_typed_result(self) -> None:
        agent = RemoteAgent("http://localhost:1234")
        result = {
            "status": "compacted",
            "messages_offloaded": 2,
            "messages_kept": 3,
            "tokens_before": 100,
            "tokens_after": 40,
            "archive_path": "/conversation_history/thread.md",
            "archive_ephemeral": False,
            "error": None,
        }
        http = SimpleNamespace(
            post=AsyncMock(
                side_effect=[
                    {
                        "status": "interrupt",
                        "request": {
                            "type": "hook_invocation",
                            "request": {"invocation_id": "hook-1"},
                        },
                    },
                    {"status": "complete", "result": result},
                ]
            )
        )
        graph = _offload_graph(http)
        fulfill = AsyncMock(return_value={"decision": "allow"})

        with patch.object(agent, "_get_graph", return_value=graph):
            actual = await agent.aoffload(
                config={"configurable": {"thread_id": "thread"}},
                context={"model": "test:model"},
                fulfill_hook=fulfill,
            )

        assert actual == result
        assert http.post.await_count == 2
        first = http.post.await_args_list[0].kwargs["json"]
        second = http.post.await_args_list[1].kwargs["json"]
        assert first["context"] == {
            "model": "test:model",
            "workspace": {"workspace_id": "test-workspace"},
        }
        assert "messages" not in first
        assert first["operation_id"] == second["operation_id"]
        assert second["hook_responses"] == {"hook-1": {"decision": "allow"}}
        fulfill.assert_awaited_once()

    async def test_hook_interrupt_payload_round_trips_from_the_server(self) -> None:
        """A real server-built interrupt payload must survive the client's parse.

        Uses `build_hook_interrupt_payload` output rather than a hand-written
        dict, and feeds the client's reply back through the server-side lookup
        key, so a payload-field rename or a UUID/str key mismatch fails here
        instead of breaking `/offload` only for users with hooks configured.
        """
        from datetime import UTC, datetime, timedelta
        from pathlib import Path
        from uuid import uuid4

        from deepagents_code.hooks.interrupt import build_hook_interrupt_payload
        from deepagents_code.hooks.models.domain import (
            ApprovalMode,
            HookContext,
            HookEvent,
            HookInvocation,
            PreCompactEvent,
        )
        from deepagents_code.hooks.models.transport import HookInvocationRequest

        invocation_id = uuid4()
        request = HookInvocationRequest(
            protocol_version=1,
            invocation_id=invocation_id,
            snapshot_id="snapshot-1",
            run_id="run-1",
            invocation=HookInvocation(
                context=HookContext(
                    thread_id="thread",
                    cwd=Path("/tmp"),
                    approval_mode=ApprovalMode.MANUAL,
                ),
                event=PreCompactEvent(event=HookEvent.PRE_COMPACT, trigger="manual"),
            ),
            deadline=datetime.now(UTC) + timedelta(seconds=60),
        )
        payload = build_hook_interrupt_payload(request)

        agent = RemoteAgent("http://localhost:1234")
        result = {
            "status": "compacted",
            "messages_offloaded": 1,
            "messages_kept": 1,
            "tokens_before": 10,
            "tokens_after": 5,
            "archive_path": "/conversation_history/thread.md",
            "archive_ephemeral": False,
            "error": None,
        }
        http = SimpleNamespace(
            post=AsyncMock(
                side_effect=[
                    {"status": "interrupt", "request": payload},
                    {"status": "complete", "result": result},
                ]
            )
        )
        graph = _offload_graph(http)
        fulfill = AsyncMock(return_value={"decision": "allow"})

        with patch.object(agent, "_get_graph", return_value=graph):
            actual = await agent.aoffload(
                config={"configurable": {"thread_id": "thread"}},
                context={},
                fulfill_hook=fulfill,
            )

        assert actual == result
        # The key the client accumulates must be exactly the key the server's
        # `_invoke_hook` looks up: `str(request.invocation_id)`.
        responses = http.post.await_args_list[1].kwargs["json"]["hook_responses"]
        assert responses == {str(invocation_id): {"decision": "allow"}}

    async def test_non_compacted_result_needs_no_statistics(self) -> None:
        """`empty`/`noop`/`denied` results carry no stats the renderer reads."""
        agent = RemoteAgent("http://localhost:1234")
        result = {"status": "denied", "error": "Blocked by a compaction hook"}
        http = SimpleNamespace(
            post=AsyncMock(return_value={"status": "complete", "result": result})
        )
        graph = _offload_graph(http)

        with patch.object(agent, "_get_graph", return_value=graph):
            actual = await agent.aoffload(
                config={"configurable": {"thread_id": "thread"}},
                context={},
                fulfill_hook=AsyncMock(),
            )

        assert actual == result

    async def test_round_limit_logs_the_ids_it_saw(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exhaustion must be diagnosable and must not assert a cause."""
        from deepagents_code.client.remote_client import _OFFLOAD_MAX_RESUME_ROUNDS

        agent = RemoteAgent("http://localhost:1234")
        counter = itertools.count()

        async def _always_interrupt(  # noqa: RUF029  # must be awaitable
            *_args: object, **_kwargs: object
        ) -> dict:
            return {
                "status": "interrupt",
                "request": {
                    "type": "hook_invocation",
                    "request": {"invocation_id": f"hook-{next(counter)}"},
                },
            }

        http = SimpleNamespace(post=_always_interrupt)
        graph = _offload_graph(http)

        with (
            patch.object(agent, "_get_graph", return_value=graph),
            caplog.at_level(logging.WARNING),
            pytest.raises(RuntimeError, match="hook rounds"),
        ):
            await agent.aoffload(
                config={"configurable": {"thread_id": "thread"}},
                context={},
                fulfill_hook=AsyncMock(return_value={}),
            )

        assert f"exceeded {_OFFLOAD_MAX_RESUME_ROUNDS} hook rounds" in caplog.text
        assert "hook-0" in caplog.text

    async def test_round_limit_does_not_fulfill_an_extra_hook(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The final round reads a result; it must not answer another hook.

        The loop runs `_OFFLOAD_MAX_RESUME_ROUNDS + 1` times because the extra
        iteration exists to POST the last fulfillment and read the reply. Drop
        the guarding `break` and it fulfills one hook too many while still
        reporting the lower number, so assert the count, not just the message.
        """
        from deepagents_code.client.remote_client import _OFFLOAD_MAX_RESUME_ROUNDS

        agent = RemoteAgent("http://localhost:1234")
        counter = itertools.count()

        async def _always_interrupt(  # noqa: RUF029  # must be awaitable
            *_args: object, **_kwargs: object
        ) -> dict[str, object]:
            return {
                "status": "interrupt",
                "request": {
                    "type": "hook_invocation",
                    "request": {"invocation_id": f"hook-{next(counter)}"},
                },
            }

        http = SimpleNamespace(post=_always_interrupt)
        graph = _offload_graph(http)
        fulfill = AsyncMock(return_value={})

        with (
            patch.object(agent, "_get_graph", return_value=graph),
            caplog.at_level(logging.WARNING),
            pytest.raises(RuntimeError, match="hook rounds"),
        ):
            await agent.aoffload(
                config={"configurable": {"thread_id": "thread"}},
                context={},
                fulfill_hook=fulfill,
            )

        assert fulfill.await_count == _OFFLOAD_MAX_RESUME_ROUNDS
