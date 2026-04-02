"""Unit tests for MicroCompactionMiddleware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.middleware.micro_compaction import (
    _FILE_READ_STUB,
    _FILE_READ_TOOLS,
    _STALE_STUB,
    MicroCompactionMiddleware,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = 1000.0  # arbitrary monotonic baseline


def _make_tool_msg(
    tool_call_id: str,
    content: str = "some result",
    name: str = "my_tool",
    *,
    msg_id: str | None = None,
    status: str | None = None,
) -> ToolMessage:
    kwargs: dict = {
        "content": content,
        "tool_call_id": tool_call_id,
        "name": name,
        "id": msg_id or tool_call_id,
    }
    if status is not None:
        kwargs["status"] = status
    return ToolMessage(**kwargs)


def _make_middleware(**kwargs: int) -> MicroCompactionMiddleware:
    return MicroCompactionMiddleware(**kwargs)


def _make_request(messages: list) -> MagicMock:
    """Stub ModelRequest that records the messages passed to override()."""
    req = MagicMock()
    req.messages = messages
    req.override.side_effect = lambda **kw: MagicMock(messages=kw["messages"])
    return req


def _make_handler() -> tuple[MagicMock, list]:
    """Return a (mock handler, captured_requests) pair."""
    captured: list[MagicMock] = []
    resp = MagicMock()

    def _handler(request: MagicMock) -> MagicMock:
        captured.append(request)
        return resp

    handler = MagicMock(side_effect=_handler)
    return handler, captured


# ---------------------------------------------------------------------------
# _compact_messages — core logic
# ---------------------------------------------------------------------------


class TestCompactMessages:
    def test_stale_tool_message_content_replaced(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        msg = _make_tool_msg("c1")

        # First call: registers the message at _NOW
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])

        # Second call: 301 seconds later — message is now stale
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 301):
            result = mw._compact_messages([msg])

        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)
        assert result[0].content == _STALE_STUB

    def test_fresh_tool_message_content_preserved(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        msg = _make_tool_msg("c2", content="important data")

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])

        # Only 100 seconds later — not stale yet
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 100):
            result = mw._compact_messages([msg])

        assert result[0].content == "important data"

    def test_preserve_recent_keeps_last_n_messages(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=3)

        old_msgs = [_make_tool_msg(f"old-{i}") for i in range(4)]
        recent_msgs = [_make_tool_msg(f"recent-{i}") for i in range(3)]
        all_msgs = old_msgs + recent_msgs

        # Register all at _NOW
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages(all_msgs)

        # 400 seconds later: all messages are old, but the last 3 are "recent"
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 400):
            result = mw._compact_messages(all_msgs)

        # First 4 messages should be compacted
        for r in result[:4]:
            assert r.content == _STALE_STUB, f"expected stub, got: {r.content!r}"
        # Last 3 are preserved
        for orig, r in zip(recent_msgs, result[4:], strict=True):
            assert r.content == orig.content

    def test_error_results_always_preserved(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        msg = _make_tool_msg("err1", content="error details", status="error")

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 500):
            result = mw._compact_messages([msg])

        assert result[0].content == "error details"

    def test_file_read_tools_use_specific_stub(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)

        for tool_name in _FILE_READ_TOOLS:
            msg = _make_tool_msg(f"fr-{tool_name}", name=tool_name, msg_id=f"fr-{tool_name}")
            with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
                mw._compact_messages([msg])
            with patch(
                "deepagents.middleware.micro_compaction.time.monotonic",
                return_value=_NOW + 400,
            ):
                result = mw._compact_messages([msg])
            assert result[0].content == _FILE_READ_STUB, f"tool={tool_name!r}"

    def test_non_file_read_tools_use_generic_stub(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        msg = _make_tool_msg("exec1", name="execute")

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 400):
            result = mw._compact_messages([msg])

        assert result[0].content == _STALE_STUB

    def test_non_tool_messages_not_modified(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        human = HumanMessage(content="hello", id="h1")
        ai = AIMessage(content="world", id="a1")

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([human, ai])
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 500):
            result = mw._compact_messages([human, ai])

        assert result[0].content == "hello"
        assert result[1].content == "world"

    def test_tool_message_without_id_not_compacted(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        # ToolMessage with no id cannot be tracked by timestamp
        msg = ToolMessage(content="untracked", tool_call_id="x", name="my_tool")

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 500):
            result = mw._compact_messages([msg])

        assert result[0].content == "untracked"

    def test_compacted_message_preserves_structure(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        msg = _make_tool_msg("c-struct", content="large output", name="execute", msg_id="id-struct")

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 400):
            (result,) = mw._compact_messages([msg])

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "c-struct"
        assert result.name == "execute"
        assert result.id == "id-struct"
        assert result.content == _STALE_STUB

    def test_message_not_compacted_on_first_observation(self) -> None:
        """A message first seen right now should never be immediately compacted."""
        mw = _make_middleware(max_age_seconds=0, preserve_recent=0)
        msg = _make_tool_msg("c-new")

        # max_age_seconds=0 but elapsed == 0 on first observation
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            result = mw._compact_messages([msg])

        # elapsed == 0, threshold is 0: condition is strictly > so not compacted
        assert result[0].content == "some result"

    def test_second_observation_same_time_not_compacted(self) -> None:
        """If re-observed at the same monotonic time, message is not yet stale."""
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        msg = _make_tool_msg("c-sametime")

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])
            result = mw._compact_messages([msg])

        assert result[0].content == "some result"


# ---------------------------------------------------------------------------
# wrap_model_call / awrap_model_call
# ---------------------------------------------------------------------------


class TestWrapModelCall:
    def test_wrap_model_call_passes_compacted_messages_to_handler(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        msg = _make_tool_msg("wmc1")
        request = _make_request([msg])
        handler, captured = _make_handler()

        # Register message, then advance time
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 400):
            mw.wrap_model_call(request, handler)

        assert handler.called
        passed_messages = captured[0].messages
        assert passed_messages[0].content == _STALE_STUB

    def test_wrap_model_call_uses_override(self) -> None:
        mw = _make_middleware()
        msg = HumanMessage(content="hi")
        request = _make_request([msg])
        handler = MagicMock(return_value=MagicMock())

        mw.wrap_model_call(request, handler)

        request.override.assert_called_once()

    async def test_awrap_model_call_passes_compacted_messages_to_handler(self) -> None:
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        msg = _make_tool_msg("awmc1")
        request = _make_request([msg])
        captured: list[MagicMock] = []

        async def async_handler(req: MagicMock) -> MagicMock:
            captured.append(req)
            return MagicMock()

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 400):
            await mw.awrap_model_call(request, async_handler)

        assert len(captured) == 1
        assert captured[0].messages[0].content == _STALE_STUB

    async def test_awrap_model_call_uses_override(self) -> None:
        mw = _make_middleware()
        msg = HumanMessage(content="async hi")
        request = _make_request([msg])

        async def async_handler(req: MagicMock) -> MagicMock:  # noqa: ARG001
            return MagicMock()

        await mw.awrap_model_call(request, async_handler)

        request.override.assert_called_once()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_message_list(self) -> None:
        mw = _make_middleware()
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            result = mw._compact_messages([])
        assert result == []

    def test_preserve_recent_larger_than_message_list(self) -> None:
        """preserve_recent > len(messages): all messages kept."""
        mw = _make_middleware(max_age_seconds=300, preserve_recent=100)
        msgs = [_make_tool_msg(f"big-{i}") for i in range(5)]

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages(msgs)
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 400):
            result = mw._compact_messages(msgs)

        for orig, r in zip(msgs, result, strict=True):
            assert r.content == orig.content

    def test_seen_at_not_overwritten_on_subsequent_calls(self) -> None:
        """Once a message is registered, its timestamp is not updated."""
        mw = _make_middleware(max_age_seconds=300, preserve_recent=0)
        msg = _make_tool_msg("sticky")

        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW):
            mw._compact_messages([msg])

        # Fake a later call where it re-appears but still within window
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 200):
            mw._compact_messages([msg])  # re-observes but should not update timestamp

        # Now 301 seconds after original observation
        with patch("deepagents.middleware.micro_compaction.time.monotonic", return_value=_NOW + 301):
            result = mw._compact_messages([msg])

        # Elapsed is 301 from first observation => should be compacted
        assert result[0].content == _STALE_STUB
