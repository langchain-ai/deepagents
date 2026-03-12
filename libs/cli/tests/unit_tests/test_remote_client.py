"""Tests for RemoteAgent, _to_uuid, _convert_message_data, and helpers."""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from deepagents_cli.remote_client import (
    RemoteAgent,
    _convert_ai_message,
    _convert_human_message,
    _convert_interrupts,
    _convert_message_data,
    _convert_tool_message,
    _prepare_config,
    _to_uuid,
)

# ---------------------------------------------------------------------------
# _to_uuid
# ---------------------------------------------------------------------------


class TestToUuid:
    def test_short_id_becomes_valid_uuid(self) -> None:
        result = _to_uuid("461bc7c2")
        uuid.UUID(result)
        assert "0000-0000-0000-000000000000" not in result

    def test_full_uuid_passthrough(self) -> None:
        full = str(uuid.uuid4())
        assert _to_uuid(full) == full

    def test_deterministic(self) -> None:
        assert _to_uuid("abcd1234") == _to_uuid("abcd1234")


# ---------------------------------------------------------------------------
# _prepare_config
# ---------------------------------------------------------------------------


class TestPrepareConfig:
    def test_converts_thread_id_to_uuid(self) -> None:
        config = {"configurable": {"thread_id": "short"}}
        result = _prepare_config(config)
        uuid.UUID(result["configurable"]["thread_id"])

    def test_preserves_valid_uuid(self) -> None:
        tid = str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}
        result = _prepare_config(config)
        assert result["configurable"]["thread_id"] == tid

    def test_none_config(self) -> None:
        result = _prepare_config(None)
        assert result == {"configurable": {}}

    def test_does_not_mutate_original(self) -> None:
        config = {"configurable": {"thread_id": "abc"}}
        _prepare_config(config)
        assert config["configurable"]["thread_id"] == "abc"

    def test_missing_configurable_key(self) -> None:
        result = _prepare_config({"other": "value"})
        assert result["configurable"] == {}

    def test_empty_string_thread_id_not_converted(self) -> None:
        result = _prepare_config({"configurable": {"thread_id": ""}})
        assert result["configurable"]["thread_id"] == ""


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
    events: list[tuple[tuple[str, ...], str, Any]],
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
    return {"configurable": {"thread_id": "t1"}}


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

        await agent.aget_state({"configurable": {"thread_id": "short"}})
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
            {"configurable": {"thread_id": "short"}}, {"key": "val"}
        )
        call_config = mock_graph.aupdate_state.call_args[0][0]
        uuid.UUID(call_config["configurable"]["thread_id"])


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
