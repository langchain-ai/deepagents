"""Unit tests for QueueLookaheadMiddleware."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, "tests/unit_tests")
from chat_model import GenericFakeChatModel

from deepagents.middleware.queue_lookahead import (
    QueueLookaheadMiddleware,
    _convert_to_human_messages,
    _extract_messages_from_run,
)

_PATCH_GET_THREAD_ID = "deepagents.middleware.queue_lookahead._get_thread_id"
_PATCH_GET_DEFAULT_CLIENT = "deepagents.middleware.queue_lookahead._get_default_client"


# ---------------------------------------------------------------------------
# Helper extraction tests
# ---------------------------------------------------------------------------


class TestExtractMessagesFromRun:
    """Tests for _extract_messages_from_run helper."""

    def test_extracts_messages_from_kwargs(self):
        run = {
            "run_id": "r1",
            "kwargs": {
                "input": {
                    "messages": [{"role": "user", "content": "hello"}],
                },
            },
        }
        result = _extract_messages_from_run(run)
        assert result == [{"role": "user", "content": "hello"}]

    def test_empty_when_no_kwargs(self):
        run = {"run_id": "r1"}
        assert _extract_messages_from_run(run) == []

    def test_empty_when_no_input(self):
        run = {"run_id": "r1", "kwargs": {}}
        assert _extract_messages_from_run(run) == []

    def test_empty_when_no_messages(self):
        run = {"run_id": "r1", "kwargs": {"input": {}}}
        assert _extract_messages_from_run(run) == []

    def test_empty_when_input_is_not_dict(self):
        run = {"run_id": "r1", "kwargs": {"input": "string"}}
        assert _extract_messages_from_run(run) == []

    def test_multiple_messages(self):
        run = {
            "run_id": "r1",
            "kwargs": {
                "input": {
                    "messages": [
                        {"role": "user", "content": "first"},
                        {"role": "user", "content": "second"},
                    ],
                },
            },
        }
        result = _extract_messages_from_run(run)
        assert len(result) == 2


class TestConvertToHumanMessages:
    """Tests for _convert_to_human_messages helper."""

    def test_converts_user_messages(self):
        raw = [{"role": "user", "content": "hello"}]
        result = _convert_to_human_messages(raw)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hello"

    def test_converts_human_role(self):
        raw = [{"role": "human", "content": "hello"}]
        result = _convert_to_human_messages(raw)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_ignores_non_user_messages(self):
        raw = [
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "real message"},
        ]
        result = _convert_to_human_messages(raw)
        assert len(result) == 1
        assert result[0].content == "real message"

    def test_ignores_empty_content(self):
        raw = [{"role": "user", "content": ""}]
        result = _convert_to_human_messages(raw)
        assert len(result) == 0

    def test_empty_input(self):
        assert _convert_to_human_messages([]) == []


# ---------------------------------------------------------------------------
# Middleware construction tests
# ---------------------------------------------------------------------------


class TestMiddlewareConstruction:
    """Tests for QueueLookaheadMiddleware initialization."""

    def test_default_construction(self):
        """Can be constructed with no args (client lazily created)."""
        mw = QueueLookaheadMiddleware()
        assert mw._client is None
        assert mw._cancel_action == "interrupt"

    def test_explicit_client(self):
        client = MagicMock()
        mw = QueueLookaheadMiddleware(client=client)
        assert mw._client is client

    def test_custom_cancel_action(self):
        mw = QueueLookaheadMiddleware(cancel_action="rollback")
        assert mw._cancel_action == "rollback"

    def test_lazy_client_resolution(self):
        """Client is created via _get_default_client on first access."""
        mock_client = MagicMock()
        mw = QueueLookaheadMiddleware()
        with patch(_PATCH_GET_DEFAULT_CLIENT, return_value=mock_client):
            resolved = mw._resolved_client
            assert resolved is mock_client
            # Second access returns cached client
            assert mw._resolved_client is mock_client


# ---------------------------------------------------------------------------
# abefore_model tests
# ---------------------------------------------------------------------------


def _make_state(messages: list | None = None) -> dict[str, Any]:
    """Create a minimal agent state dict.

    Args:
        messages: Messages in the state.

    Returns:
        Agent state dict.
    """
    return {"messages": messages or [HumanMessage(content="original")]}


def _make_runtime() -> MagicMock:
    """Create a mock Runtime."""
    return MagicMock()


class TestAbeforeModel:
    """Tests for abefore_model hook."""

    async def test_no_thread_id_returns_none(self):
        """When no thread_id is available, returns None (no state update)."""
        with patch(_PATCH_GET_THREAD_ID, return_value=None):
            client = MagicMock()
            mw = QueueLookaheadMiddleware(client=client)

            result = await mw.abefore_model(_make_state(), _make_runtime())

            assert result is None

    async def test_no_pending_runs_returns_none(self):
        """When there are no pending runs, returns None."""
        with patch(_PATCH_GET_THREAD_ID, return_value="thread-1"):
            client = MagicMock()
            client.runs = MagicMock()
            client.runs.list = AsyncMock(return_value=[])
            mw = QueueLookaheadMiddleware(client=client)

            result = await mw.abefore_model(_make_state(), _make_runtime())

            client.runs.list.assert_called_once_with(thread_id="thread-1", status="pending")
            assert result is None

    async def test_returns_messages_state_update(self):
        """Pending run messages are returned as a state update."""
        with patch(_PATCH_GET_THREAD_ID, return_value="thread-1"):
            client = MagicMock()
            client.runs = MagicMock()
            client.runs.list = AsyncMock(
                return_value=[
                    {
                        "run_id": "pending-run-1",
                        "kwargs": {
                            "input": {
                                "messages": [{"role": "user", "content": "follow-up"}],
                            },
                        },
                    },
                ]
            )
            client.runs.cancel = AsyncMock()

            mw = QueueLookaheadMiddleware(client=client)
            result = await mw.abefore_model(_make_state(), _make_runtime())

            assert result is not None
            assert "messages" in result
            assert len(result["messages"]) == 1
            assert isinstance(result["messages"][0], HumanMessage)
            assert result["messages"][0].content == "follow-up"

            # Verify pending run was cancelled
            client.runs.cancel.assert_called_once_with(
                thread_id="thread-1",
                run_id="pending-run-1",
                action="interrupt",
            )

    async def test_multiple_pending_runs(self):
        """Multiple pending runs are all consumed and returned."""
        with patch(_PATCH_GET_THREAD_ID, return_value="thread-1"):
            client = MagicMock()
            client.runs = MagicMock()
            client.runs.list = AsyncMock(
                return_value=[
                    {
                        "run_id": "run-1",
                        "kwargs": {"input": {"messages": [{"role": "user", "content": "msg1"}]}},
                    },
                    {
                        "run_id": "run-2",
                        "kwargs": {"input": {"messages": [{"role": "user", "content": "msg2"}]}},
                    },
                ]
            )
            client.runs.cancel = AsyncMock()

            mw = QueueLookaheadMiddleware(client=client)
            result = await mw.abefore_model(_make_state(), _make_runtime())

            # Both runs cancelled
            assert client.runs.cancel.call_count == 2

            # Messages from both runs returned
            assert result is not None
            assert len(result["messages"]) == 2
            assert result["messages"][0].content == "msg1"
            assert result["messages"][1].content == "msg2"

    async def test_list_failure_returns_none(self):
        """If listing pending runs fails, returns None."""
        with patch(_PATCH_GET_THREAD_ID, return_value="thread-1"):
            client = MagicMock()
            client.runs = MagicMock()
            client.runs.list = AsyncMock(side_effect=Exception("network error"))

            mw = QueueLookaheadMiddleware(client=client)
            result = await mw.abefore_model(_make_state(), _make_runtime())

            assert result is None

    async def test_cancel_failure_still_returns_messages(self):
        """If cancelling a run fails, messages are still returned."""
        with patch(_PATCH_GET_THREAD_ID, return_value="thread-1"):
            client = MagicMock()
            client.runs = MagicMock()
            client.runs.list = AsyncMock(
                return_value=[
                    {
                        "run_id": "run-1",
                        "kwargs": {"input": {"messages": [{"role": "user", "content": "msg"}]}},
                    },
                ]
            )
            client.runs.cancel = AsyncMock(side_effect=Exception("cancel failed"))

            mw = QueueLookaheadMiddleware(client=client)
            result = await mw.abefore_model(_make_state(), _make_runtime())

            assert result is not None
            assert len(result["messages"]) == 1

    async def test_filters_non_user_messages(self):
        """Only user/human messages from pending runs are returned."""
        with patch(_PATCH_GET_THREAD_ID, return_value="thread-1"):
            client = MagicMock()
            client.runs = MagicMock()
            client.runs.list = AsyncMock(
                return_value=[
                    {
                        "run_id": "run-1",
                        "kwargs": {
                            "input": {
                                "messages": [
                                    {"role": "system", "content": "sys msg"},
                                    {"role": "user", "content": "user msg"},
                                    {"role": "assistant", "content": "ai msg"},
                                ],
                            },
                        },
                    },
                ]
            )
            client.runs.cancel = AsyncMock()

            mw = QueueLookaheadMiddleware(client=client)
            result = await mw.abefore_model(_make_state(), _make_runtime())

            assert result is not None
            assert len(result["messages"]) == 1
            assert result["messages"][0].content == "user msg"

    async def test_custom_cancel_action(self):
        """Custom cancel_action is passed to runs.cancel."""
        with patch(_PATCH_GET_THREAD_ID, return_value="thread-1"):
            client = MagicMock()
            client.runs = MagicMock()
            client.runs.list = AsyncMock(
                return_value=[
                    {
                        "run_id": "run-1",
                        "kwargs": {"input": {"messages": [{"role": "user", "content": "msg"}]}},
                    },
                ]
            )
            client.runs.cancel = AsyncMock()

            mw = QueueLookaheadMiddleware(client=client, cancel_action="rollback")
            result = await mw.abefore_model(_make_state(), _make_runtime())

            assert result is not None
            client.runs.cancel.assert_called_once_with(
                thread_id="thread-1",
                run_id="run-1",
                action="rollback",
            )

    async def test_default_client_used_when_none_provided(self):
        """When no client is provided, _get_default_client is called lazily."""
        mock_client = MagicMock()
        mock_client.runs = MagicMock()
        mock_client.runs.list = AsyncMock(return_value=[])

        with (
            patch(_PATCH_GET_THREAD_ID, return_value="thread-1"),
            patch(_PATCH_GET_DEFAULT_CLIENT, return_value=mock_client),
        ):
            mw = QueueLookaheadMiddleware()
            result = await mw.abefore_model(_make_state(), _make_runtime())

            mock_client.runs.list.assert_called_once_with(thread_id="thread-1", status="pending")
            assert result is None


# ---------------------------------------------------------------------------
# Integration tests: real graph with middleware
# ---------------------------------------------------------------------------


def _make_pending_run(run_id: str, content: str) -> dict[str, Any]:
    """Build a fake pending run dict.

    Args:
        run_id: The run ID.
        content: The user message content.

    Returns:
        A dict shaped like a LangGraph SDK Run.
    """
    return {
        "run_id": run_id,
        "kwargs": {"input": {"messages": [{"role": "user", "content": content}]}},
    }


class TestGraphIntegration:
    """Tests that compile a real agent graph with QueueLookaheadMiddleware."""

    async def test_pending_messages_reach_model(self):
        """Pending messages injected by before_model are seen by the model."""
        mock_client = MagicMock()
        mock_client.runs = MagicMock()
        # First call to before_model finds a pending run; subsequent calls find none
        mock_client.runs.list = AsyncMock(
            side_effect=[
                [_make_pending_run("run-1", "actually, use Python 3.12")],
                [],  # no more pending on the second model call (if any)
            ]
        )
        mock_client.runs.cancel = AsyncMock()

        mw = QueueLookaheadMiddleware(client=mock_client)
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Got it, using 3.12.")]))

        agent = create_agent(model=model, middleware=[mw])

        config = {"configurable": {"thread_id": "test-thread-1"}}
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Write me a Python script")]},
            config,
        )

        # The model should have seen both the original message and the injected one
        assert len(model.call_history) >= 1
        model_messages = model.call_history[0]["messages"]
        contents = [m.content for m in model_messages if isinstance(m, HumanMessage)]
        assert "Write me a Python script" in contents
        assert "actually, use Python 3.12" in contents

        # The injected message should appear in the final state
        final_messages = result["messages"]
        human_contents = [m.content for m in final_messages if isinstance(m, HumanMessage)]
        assert "actually, use Python 3.12" in human_contents

        # The pending run was cancelled
        mock_client.runs.cancel.assert_called_once_with(
            thread_id="test-thread-1",
            run_id="run-1",
            action="interrupt",
        )

    async def test_pending_messages_checkpointed(self):
        """Pending messages are persisted to the checkpoint before the model runs."""
        mock_client = MagicMock()
        mock_client.runs = MagicMock()
        mock_client.runs.list = AsyncMock(
            side_effect=[
                [_make_pending_run("run-1", "checkpoint me")],
                [],
            ]
        )
        mock_client.runs.cancel = AsyncMock()

        mw = QueueLookaheadMiddleware(client=mock_client)
        checkpointer = InMemorySaver()
        model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))

        agent = create_agent(model=model, middleware=[mw], checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-checkpoint"}}
        await agent.ainvoke(
            {"messages": [HumanMessage(content="original")]},
            config,
        )

        # Verify the injected message is in the checkpointed state
        state = agent.get_state(config)
        messages = state.values["messages"]
        human_contents = [m.content for m in messages if isinstance(m, HumanMessage)]
        assert "checkpoint me" in human_contents

    async def test_no_pending_messages_passthrough(self):
        """When there are no pending runs, the agent behaves normally."""
        mock_client = MagicMock()
        mock_client.runs = MagicMock()
        mock_client.runs.list = AsyncMock(return_value=[])

        mw = QueueLookaheadMiddleware(client=mock_client)
        model = GenericFakeChatModel(messages=iter([AIMessage(content="normal response")]))

        agent = create_agent(model=model, middleware=[mw])

        config = {"configurable": {"thread_id": "test-passthrough"}}
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="hello")]},
            config,
        )

        # Model was called with just the original message
        model_messages = model.call_history[0]["messages"]
        human_messages = [m for m in model_messages if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        assert human_messages[0].content == "hello"

        # Response is normal
        assert any(m.content == "normal response" for m in result["messages"])

    async def test_multiple_pending_runs_in_graph(self):
        """Multiple pending runs are all drained and injected in one before_model pass."""
        mock_client = MagicMock()
        mock_client.runs = MagicMock()
        mock_client.runs.list = AsyncMock(
            side_effect=[
                [
                    _make_pending_run("run-1", "correction 1"),
                    _make_pending_run("run-2", "correction 2"),
                ],
                [],
            ]
        )
        mock_client.runs.cancel = AsyncMock()

        mw = QueueLookaheadMiddleware(client=mock_client)
        model = GenericFakeChatModel(messages=iter([AIMessage(content="understood both")]))

        agent = create_agent(model=model, middleware=[mw])

        config = {"configurable": {"thread_id": "test-multi"}}
        await agent.ainvoke(
            {"messages": [HumanMessage(content="do something")]},
            config,
        )

        # Model should see all three messages
        model_messages = model.call_history[0]["messages"]
        human_contents = [m.content for m in model_messages if isinstance(m, HumanMessage)]
        assert "do something" in human_contents
        assert "correction 1" in human_contents
        assert "correction 2" in human_contents

        # Both pending runs cancelled
        assert mock_client.runs.cancel.call_count == 2


# ---------------------------------------------------------------------------
# Export test
# ---------------------------------------------------------------------------


def test_importable_from_package():
    """QueueLookaheadMiddleware should be importable from the middleware package."""
    from deepagents.middleware import QueueLookaheadMiddleware as Imported  # noqa: PLC0415

    assert Imported is QueueLookaheadMiddleware
