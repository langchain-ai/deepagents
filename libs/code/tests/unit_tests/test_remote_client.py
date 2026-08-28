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
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from deepagents_code._env_vars import LANGSMITH_REPLICA_PROJECTS
from deepagents_code.client.remote_client import (
    RemoteAgent,
    _cancelled_tool_messages,
    _convert_ai_message,
    _convert_human_message,
    _convert_interrupts,
    _convert_message_data,
    _convert_tool_message,
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
    def test_preserves_thread_id(self) -> None:
        config = {"configurable": {"thread_id": _TEST_THREAD_ID}}
        result = _prepare_config(config)
        assert result["configurable"]["thread_id"] == _TEST_THREAD_ID

    def test_none_config(self) -> None:
        result = _prepare_config(None)
        assert result == {"configurable": {}}

    def test_does_not_mutate_original(self) -> None:
        tid = str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}
        _prepare_config(config)
        assert config["configurable"]["thread_id"] == tid

    def test_missing_configurable_key(self) -> None:
        result = _prepare_config({"other": "value"})
        assert result["configurable"] == {}

    def test_empty_string_thread_id_not_converted(self) -> None:
        result = _prepare_config({"configurable": {"thread_id": ""}})
        assert result["configurable"]["thread_id"] == ""

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
    def test_ai_message_text(self) -> None:
        msg = _convert_message_data({"type": "ai", "content": "Hello", "id": "m1"})
        assert isinstance(msg, AIMessageChunk)
        assert msg.content == "Hello"
        assert msg.id == "m1"

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

    def test_ai_message_with_dict_args_tool_calls(self) -> None:
        msg = _convert_message_data(
            {
                "type": "ai",
                "content": "",
                "id": "m1",
                "tool_calls": [{"name": "search", "args": {"q": "test"}, "id": "tc1"}],
            }
        )
        assert isinstance(msg, AIMessageChunk)
        assert msg.tool_calls[0]["name"] == "search"

    def test_ai_message_usage_metadata(self) -> None:
        msg = _convert_message_data(
            {
                "type": "ai",
                "content": "",
                "id": "m1",
                "usage_metadata": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )
        assert msg.usage_metadata["input_tokens"] == 10

    def test_ai_message_type_alias(self) -> None:
        msg = _convert_message_data({"type": "AIMessage", "content": "Hi", "id": "m1"})
        assert isinstance(msg, AIMessageChunk)
        assert msg.content == "Hi"

    def test_human_message(self) -> None:
        msg = _convert_message_data({"type": "human", "content": "Hi", "id": "m1"})
        assert isinstance(msg, HumanMessage)
        assert msg.content == "Hi"

    def test_human_message_type_alias(self) -> None:
        msg = _convert_message_data(
            {"type": "HumanMessage", "content": "Hey", "id": "m1"}
        )
        assert isinstance(msg, HumanMessage)
        assert msg.content == "Hey"

    def test_tool_message(self) -> None:
        msg = _convert_message_data(
            {
                "type": "tool",
                "content": "Sunny",
                "tool_call_id": "tc1",
                "name": "weather",
                "id": "m2",
            }
        )
        assert isinstance(msg, ToolMessage)
        assert msg.content == "Sunny"
        assert msg.tool_call_id == "tc1"

    def test_tool_message_type_alias(self) -> None:
        msg = _convert_message_data(
            {
                "type": "ToolMessage",
                "content": "result",
                "tool_call_id": "tc1",
                "name": "search",
                "id": "m3",
            }
        )
        assert isinstance(msg, ToolMessage)
        assert msg.content == "result"

    def test_tool_message_defaults(self) -> None:
        msg = _convert_message_data({"type": "tool", "id": "m1"})
        assert isinstance(msg, ToolMessage)
        assert msg.content == ""
        assert msg.tool_call_id == ""
        assert msg.name == ""
        assert msg.status == "success"

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

    def test_unknown_type_returns_none(self) -> None:
        assert _convert_message_data({"type": "unknown"}) is None


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

    def test_non_list_wraps_value(self) -> None:
        assert _convert_interrupts("not a list") == ["not a list"]

    def test_none_returns_empty(self) -> None:
        assert _convert_interrupts(None) == []

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

    async def test_no_kwarg_when_unset(self, monkeypatch) -> None:
        """Without a replica, `langsmith_tracing` is not passed at all."""
        monkeypatch.delenv(LANGSMITH_REPLICA_PROJECTS, raising=False)
        agent, captured = _make_capturing_agent()
        async for _ in agent.astream({"messages": []}, config=_config()):
            pass
        assert "langsmith_tracing" not in captured

    def test_sdk_accepts_langsmith_tracing(self) -> None:
        """The real SDK still accepts the kwarg name and `project_name` shape.

        `RemoteGraph.astream` forwards unknown kwargs to `client.runs.stream`, so
        a silent drop would happen if either the parameter or the payload key
        were renamed upstream. This guards both.
        """
        import inspect

        from langgraph_sdk.client import RunsClient
        from langgraph_sdk.schema import LangSmithTracing

        params = inspect.signature(RunsClient.stream).parameters
        assert "langsmith_tracing" in params
        assert "project_name" in LangSmithTracing.__annotations__


# ---------------------------------------------------------------------------
# RemoteAgent — astream delegation
# ---------------------------------------------------------------------------


class TestRemoteAgentAstream:
    async def test_text_message_converted(self) -> None:
        """Messages-tuple text chunks are converted to AIMessageChunk."""
        events = [((), "messages", ({"type": "ai", "content": "Hi", "id": "m1"}, {}))]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert len(results) == 1
        ns, mode, (msg, _meta) = results[0]
        assert ns == ()
        assert mode == "messages"
        assert isinstance(msg, AIMessageChunk)
        assert msg.content == "Hi"

    async def test_tool_message_converted(self) -> None:
        """Tool messages are converted to ToolMessage."""
        events = [
            (
                (),
                "messages",
                (
                    {
                        "type": "tool",
                        "content": "Sunny",
                        "tool_call_id": "tc1",
                        "name": "weather",
                        "id": "m2",
                    },
                    {},
                ),
            )
        ]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert len(results) == 1
        assert isinstance(results[0][2][0], ToolMessage)

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

    async def test_namespace_preserved(self) -> None:
        """Namespace from RemoteGraph is preserved in output."""
        events = [
            (
                ("sub", "inner"),
                "messages",
                ({"type": "ai", "content": "Hi", "id": "m1"}, {}),
            )
        ]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert results[0][0] == ("sub", "inner")

    async def test_unknown_message_type_skipped(self) -> None:
        """Unknown message types don't produce output."""
        events = [((), "messages", ({"type": "unknown", "content": "?"}, {}))]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert results == []

    async def test_missing_thread_id_raises(self) -> None:
        """Raises ValueError if thread_id is missing."""
        agent = _make_agent([])
        with pytest.raises(ValueError, match="thread_id"):
            async for _ in agent.astream({"messages": []}, config={"configurable": {}}):
                pass

    async def test_rapid_streaming(self) -> None:
        """Many rapid text events all arrive (no dropped tokens)."""
        events = [
            (
                (),
                "messages",
                ({"type": "ai", "content": f"tok{i}", "id": "m1"}, {}),
            )
            for i in range(100)
        ]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        combined = "".join(r[2][0].content for r in results)
        assert combined == "".join(f"tok{i}" for i in range(100))
        assert len(results) == 100

    async def test_non_dict_message_object_passed_through(self) -> None:
        """Pre-deserialized LangChain message objects are yielded as-is."""
        chunk = AIMessageChunk(content="pre-built", id="m1")
        events = [((), "messages", (chunk, {"run_id": "r1"}))]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert len(results) == 1
        assert results[0][2][0] is chunk
        assert results[0][2][1] == {"run_id": "r1"}

    async def test_meta_none_defaults_to_empty_dict(self) -> None:
        """None metadata is normalized to empty dict."""
        events = [((), "messages", ({"type": "ai", "content": "x", "id": "m1"}, None))]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert results[0][2][1] == {}

    async def test_unknown_mode_passed_through(self) -> None:
        """Events with unknown modes are yielded unchanged."""
        events = [((), "values", {"key": "val"})]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert len(results) == 1
        assert results[0] == ((), "values", {"key": "val"})

    async def test_non_dict_updates_falls_through(self) -> None:
        """Non-dict updates data passes through the generic yield."""
        events = [((), "updates", "string_data")]
        agent = _make_agent(events)
        results = [
            item async for item in agent.astream({"messages": []}, config=_config())
        ]
        assert len(results) == 1
        assert results[0] == ((), "updates", "string_data")


# ---------------------------------------------------------------------------
# RemoteAgent — aget_state
# ---------------------------------------------------------------------------


class TestRemoteAgentGetState:
    async def test_returns_state_on_success(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        state = MagicMock(values={"messages": []}, next=())
        mock_graph.aget_state = AsyncMock(return_value=state)
        agent._graph = mock_graph

        result = await agent.aget_state(_config())
        assert result is state

    async def test_raises_when_thread_id_missing(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        with pytest.raises(ValueError, match="thread_id"):
            await agent.aget_state({"configurable": {}})

    async def test_returns_none_on_not_found(self) -> None:
        from langgraph_sdk.errors import NotFoundError

        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        request = MagicMock()
        response = MagicMock(status_code=404, headers={})
        exc = NotFoundError("not found", response=response, body=None)
        exc.request = request
        mock_graph.aget_state = AsyncMock(side_effect=exc)
        agent._graph = mock_graph

        result = await agent.aget_state(_config())
        assert result is None

    async def test_propagates_non_404_exception(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(side_effect=ConnectionError("down"))
        agent._graph = mock_graph

        with pytest.raises(ConnectionError, match="down"):
            await agent.aget_state(_config())

    async def test_normalizes_config(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=None)
        agent._graph = mock_graph

        await agent.aget_state({"configurable": {"thread_id": _TEST_THREAD_ID}})
        call_config = mock_graph.aget_state.call_args[0][0]
        uuid.UUID(call_config["configurable"]["thread_id"])


# ---------------------------------------------------------------------------
# RemoteAgent — aupdate_state
# ---------------------------------------------------------------------------


class TestRemoteAgentUpdateState:
    async def test_delegates_to_graph(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        mock_graph.aupdate_state = AsyncMock()
        agent._graph = mock_graph

        await agent.aupdate_state(_config(), {"key": "val"})
        mock_graph.aupdate_state.assert_called_once()

    async def test_forwards_as_node(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        mock_graph.aupdate_state = AsyncMock()
        agent._graph = mock_graph

        await agent.aupdate_state(_config(), {"key": "val"}, as_node="model")

        mock_graph.aupdate_state.assert_awaited_once()
        update_args = mock_graph.aupdate_state.await_args
        assert update_args is not None
        assert update_args.kwargs["as_node"] == "model"

    async def test_raises_when_thread_id_missing(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        with pytest.raises(ValueError, match="thread_id"):
            await agent.aupdate_state({"configurable": {}}, {"key": "val"})

    async def test_propagates_exception(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        mock_graph.aupdate_state = AsyncMock(side_effect=ConnectionError("down"))
        agent._graph = mock_graph

        with pytest.raises(ConnectionError, match="down"):
            await agent.aupdate_state(_config(), {"key": "val"})

    async def test_normalizes_config(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        mock_graph = MagicMock()
        mock_graph.aupdate_state = AsyncMock()
        agent._graph = mock_graph

        await agent.aupdate_state(
            {"configurable": {"thread_id": _TEST_THREAD_ID}}, {"key": "val"}
        )
        call_config = mock_graph.aupdate_state.call_args[0][0]
        uuid.UUID(call_config["configurable"]["thread_id"])


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

    async def test_non_conflict_exception_does_not_retry(self) -> None:
        runs_list = AsyncMock()
        runs_cancel = AsyncMock()
        agent, mock_graph = self._agent_with_client(
            runs_list=runs_list,
            runs_cancel=runs_cancel,
            update_side_effect=[ConnectionError("down")],
        )

        with pytest.raises(ConnectionError, match="down"):
            await agent.aupdate_state(_config(), {"messages": []})
        assert mock_graph.aupdate_state.await_count == 1
        runs_list.assert_not_called()
        runs_cancel.assert_not_called()


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

    async def test_raises_when_thread_id_missing(self) -> None:
        """Rejects ensure-thread calls that omit `configurable.thread_id`."""
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")

        with pytest.raises(ValueError, match="thread_id"):
            await agent.aensure_thread({"configurable": {}})


# ---------------------------------------------------------------------------
# RemoteAgent — with_config
# ---------------------------------------------------------------------------


class TestRemoteAgentInit:
    def test_api_key_passed_to_remote_graph(self) -> None:
        """api_key kwarg is forwarded to RemoteGraph."""
        agent = RemoteAgent(
            url="http://localhost:8123",
            graph_name="agent",
            api_key="sk-test-123",
        )
        with patch("langgraph.pregel.remote.RemoteGraph") as mock_cls:
            agent._get_graph()
            mock_cls.assert_called_once_with(
                "agent",
                url="http://localhost:8123",
                api_key="sk-test-123",
                headers=None,
            )

    def test_headers_passed_to_remote_graph(self) -> None:
        """Headers kwarg is forwarded to RemoteGraph."""
        hdrs = {"Authorization": "Bearer tok", "X-Custom": "val"}
        agent = RemoteAgent(
            url="http://localhost:8123",
            graph_name="agent",
            headers=hdrs,
        )
        with patch("langgraph.pregel.remote.RemoteGraph") as mock_cls:
            agent._get_graph()
            mock_cls.assert_called_once_with(
                "agent",
                url="http://localhost:8123",
                api_key=None,
                headers=hdrs,
            )

    def test_defaults_no_auth(self) -> None:
        """Default construction passes None for api_key and headers."""
        agent = RemoteAgent(url="http://localhost:8123")
        with patch("langgraph.pregel.remote.RemoteGraph") as mock_cls:
            agent._get_graph()
            mock_cls.assert_called_once_with(
                "agent",
                url="http://localhost:8123",
                api_key=None,
                headers=None,
            )

    def test_graph_lazy_singleton(self) -> None:
        """_get_graph creates RemoteGraph once and caches it."""
        agent = RemoteAgent(url="http://localhost:8123")
        with patch("langgraph.pregel.remote.RemoteGraph") as mock_cls:
            g1 = agent._get_graph()
            g2 = agent._get_graph()
            assert g1 is g2
            mock_cls.assert_called_once()


class TestRemoteAgentWithConfig:
    def test_returns_self(self) -> None:
        agent = RemoteAgent(url="http://localhost:8123", graph_name="agent")
        assert agent.with_config({"configurable": {}}) is agent


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

    def test_empty_args_uses_class_name(self) -> None:
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException()
        assert exc.args == ()
        assert agent_error_type(exc) == "RemoteException"

    def test_dict_without_error_key_uses_class_name(self) -> None:
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException({"message": "x"})
        assert agent_error_type(exc) == "RemoteException"

    def test_non_string_error_key_uses_class_name(self) -> None:
        from langgraph.pregel.remote import RemoteException

        exc = RemoteException({"error": 500})
        assert agent_error_type(exc) == "RemoteException"

    def test_non_dict_payload_uses_class_name(self) -> None:
        assert agent_error_type(ValueError("boom")) == "ValueError"


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


class TestServerOffload:
    """The remote client transports operation data without graph state."""

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
        assert first["context"] == {"model": "test:model"}
        assert "messages" not in first
        assert first["operation_id"] == second["operation_id"]
        assert second["hook_responses"] == {"hook-1": {"decision": "allow"}}
        fulfill.assert_awaited_once()

    async def test_missing_route_names_the_cause(self) -> None:
        """A server without the route must not surface a bare "404 Not Found".

        A custom `graph_ref` server never registers dcode's HTTP app. An
        unregistered thread cannot reach here as a 404 -- the server answers
        409 for that -- so a 404 means the route is absent, and the message
        should say so and name a fix.
        """
        import httpx
        from langgraph_sdk.errors import NotFoundError

        agent = RemoteAgent("http://localhost:1234")
        request = httpx.Request("POST", "http://localhost/dcode/threads/t/offload")
        http = SimpleNamespace(
            post=AsyncMock(
                side_effect=NotFoundError(
                    "404 Not Found",
                    response=httpx.Response(404, request=request),
                    body=None,
                )
            )
        )
        graph = _offload_graph(http)

        with (
            patch.object(agent, "_get_graph", return_value=graph),
            pytest.raises(RuntimeError, match="does not provide dcode's /offload"),
        ):
            await agent.aoffload(
                config={"configurable": {"thread_id": "thread"}},
                context={},
                fulfill_hook=AsyncMock(),
            )

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

    @pytest.mark.parametrize(
        ("result", "match"),
        [
            ({"status": "compacted"}, "messages_offloaded"),
            ({**_COMPACTED_RESULT, "tokens_before": "100"}, "tokens_before"),
            ({**_COMPACTED_RESULT, "tokens_after": True}, "tokens_after"),
            ({}, "no status"),
            ("not a dict", "without a typed result"),
        ],
    )
    async def test_malformed_complete_result_is_refused(
        self, result: object, match: str
    ) -> None:
        """A drifted payload must fail naming the field, not `KeyError` later.

        The renderer indexes these fields unguarded, and the server has already
        committed the compaction by this point, so a `KeyError` here would be
        reported as "Offload failed" for work that actually succeeded.
        """
        agent = RemoteAgent("http://localhost:1234")
        http = SimpleNamespace(
            post=AsyncMock(return_value={"status": "complete", "result": result})
        )
        graph = _offload_graph(http)

        with (
            patch.object(agent, "_get_graph", return_value=graph),
            pytest.raises(RuntimeError, match=match),
        ):
            await agent.aoffload(
                config={"configurable": {"thread_id": "thread"}},
                context={},
                fulfill_hook=AsyncMock(),
            )

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
