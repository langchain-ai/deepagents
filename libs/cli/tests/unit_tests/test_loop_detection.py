"""Tests for LoopDetectionMiddleware (stateless rewrite)."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents_cli.loop_detection import (
    LOOP_WARNING_HARD,
    LOOP_WARNING_SOFT,
    LoopDetectionMiddleware,
    _count_file_edits,
    _hard_warning_already_shown,
    _soft_warning_already_shown,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ai_with_tool_calls(
    calls: list[dict[str, Any]],
    *,
    pending: bool = False,
) -> AIMessage:
    """Create an AIMessage with tool_calls."""
    tool_calls = [
        {"name": c["name"], "args": c.get("args", {}), "id": c.get("id", "id-1")}
        for c in calls
    ]
    msg = AIMessage(content="", tool_calls=tool_calls if pending else [])
    # For counting purposes, tool_calls on the message must be set
    # regardless of whether they are "pending" (for after_model guard).
    # The pending flag controls whether the last message looks like it
    # still has unprocessed tool calls.
    if not pending:
        # Put tool_calls for counting but mark them as resolved
        # by having them on a non-last message.
        msg = AIMessage(content="", tool_calls=tool_calls)
    return msg


def _make_edit_calls(file_path: str, n: int) -> list[AIMessage]:
    """Create *n* AIMessages each with one edit_file tool_call."""
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "edit_file", "args": {"file_path": file_path}, "id": f"id-{i}"}
            ],
        )
        for i in range(n)
    ]


def _make_tool_call_request(
    tool_name: str,
    args: dict[str, Any],
    messages: list[Any],
) -> Mock:
    """Create a mock ToolCallRequest."""
    request = Mock()
    request.tool_call = {"name": tool_name, "args": args, "id": "call-1"}
    request.state = {"messages": messages}
    return request


# ---------------------------------------------------------------------------
# _count_file_edits
# ---------------------------------------------------------------------------


class TestCountFileEdits:
    def test_empty_messages(self) -> None:
        assert _count_file_edits([]) == {}

    def test_no_edit_tools(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"file_path": "/a.py"}, "id": "1"}
                ],
            )
        ]
        assert _count_file_edits(messages) == {}

    def test_single_file(self) -> None:
        messages = _make_edit_calls("/a.py", 3)
        assert _count_file_edits(messages) == {"/a.py": 3}

    def test_multiple_files(self) -> None:
        messages = [
            *_make_edit_calls("/a.py", 2),
            *_make_edit_calls("/b.py", 1),
        ]
        counts = _count_file_edits(messages)
        assert counts == {"/a.py": 2, "/b.py": 1}

    def test_ignores_non_edit_tools(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "execute", "args": {"command": "ls"}, "id": "1"}],
            ),
            *_make_edit_calls("/a.py", 1),
        ]
        assert _count_file_edits(messages) == {"/a.py": 1}

    def test_handles_write_file(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "write_file", "args": {"file_path": "/a.py"}, "id": "1"}
                ],
            )
        ]
        assert _count_file_edits(messages) == {"/a.py": 1}

    def test_path_normalization(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "edit_file",
                        "args": {"file_path": "/foo/../bar/./a.py"},
                        "id": "1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "edit_file",
                        "args": {"file_path": "/bar/a.py"},
                        "id": "2",
                    }
                ],
            ),
        ]
        assert _count_file_edits(messages) == {"/bar/a.py": 2}

    def test_relative_and_absolute_resolve_to_same(self) -> None:
        """Relative path resolves to absolute, matching an absolute edit."""
        from pathlib import Path

        abs_path = str(Path("src/a.py").resolve())
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "edit_file", "args": {"file_path": "src/a.py"}, "id": "1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "edit_file", "args": {"file_path": abs_path}, "id": "2"}
                ],
            ),
        ]
        assert _count_file_edits(messages) == {abs_path: 2}

    def test_ignores_non_ai_messages(self) -> None:
        messages = [
            HumanMessage(content="hello"),
            ToolMessage(content="result", tool_call_id="1"),
            *_make_edit_calls("/a.py", 1),
        ]
        assert _count_file_edits(messages) == {"/a.py": 1}

    def test_fallback_to_path_arg(self) -> None:
        """Falls back to 'path' arg when 'file_path' is absent."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "edit_file", "args": {"path": "/a.py"}, "id": "1"}
                ],
            )
        ]
        assert _count_file_edits(messages) == {"/a.py": 1}

    def test_fallback_to_unknown(self) -> None:
        """Falls back to 'unknown' (resolved) when no path arg is present."""
        from pathlib import Path

        messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "edit_file", "args": {}, "id": "1"}],
            )
        ]
        resolved = str(Path("unknown").resolve())
        assert _count_file_edits(messages) == {resolved: 1}


# ---------------------------------------------------------------------------
# Marker deduplication helpers
# ---------------------------------------------------------------------------


class TestMarkerDetection:
    def test_soft_warning_not_shown(self) -> None:
        assert not _soft_warning_already_shown([], "/a.py")

    def test_soft_warning_shown(self) -> None:
        messages = [
            ToolMessage(
                content="OK" + LOOP_WARNING_SOFT.format(file_path="/a.py", count=4),
                tool_call_id="1",
            )
        ]
        assert _soft_warning_already_shown(messages, "/a.py")

    def test_soft_warning_different_file(self) -> None:
        messages = [
            ToolMessage(
                content="OK" + LOOP_WARNING_SOFT.format(file_path="/b.py", count=4),
                tool_call_id="1",
            )
        ]
        assert not _soft_warning_already_shown(messages, "/a.py")

    def test_hard_warning_not_shown(self) -> None:
        assert not _hard_warning_already_shown([], "/a.py")

    def test_hard_warning_shown(self) -> None:
        messages = [
            HumanMessage(content=LOOP_WARNING_HARD.format(file_path="/a.py", count=8))
        ]
        assert _hard_warning_already_shown(messages, "/a.py")

    def test_hard_warning_different_file(self) -> None:
        messages = [
            HumanMessage(content=LOOP_WARNING_HARD.format(file_path="/b.py", count=8))
        ]
        assert not _hard_warning_already_shown(messages, "/a.py")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self) -> None:
        mw = LoopDetectionMiddleware()
        assert mw.soft_threshold == 8
        assert mw.hard_threshold == 12

    def test_custom_thresholds(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=2, hard_threshold=5)
        assert mw.soft_threshold == 2
        assert mw.hard_threshold == 5

    def test_soft_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="soft_threshold must be >= 1"):
            LoopDetectionMiddleware(soft_threshold=0)

    def test_hard_not_greater_raises(self) -> None:
        with pytest.raises(
            ValueError, match="hard_threshold must be greater than soft_threshold"
        ):
            LoopDetectionMiddleware(soft_threshold=5, hard_threshold=5)

    def test_hard_less_than_soft_raises(self) -> None:
        with pytest.raises(
            ValueError, match="hard_threshold must be greater than soft_threshold"
        ):
            LoopDetectionMiddleware(soft_threshold=5, hard_threshold=3)


# ---------------------------------------------------------------------------
# No mutable state
# ---------------------------------------------------------------------------


class TestNoMutableState:
    def test_only_threshold_attrs(self) -> None:
        mw = LoopDetectionMiddleware()
        instance_vars = vars(mw)
        assert set(instance_vars.keys()) == {"soft_threshold", "hard_threshold"}


# ---------------------------------------------------------------------------
# Soft warning (wrap_tool_call)
# ---------------------------------------------------------------------------


class TestSoftWarning:
    def test_no_warning_below_threshold(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=4, hard_threshold=8)
        # 2 prior edits + 1 current = 3, below threshold of 4
        messages = _make_edit_calls("/a.py", 2)
        request = _make_tool_call_request("edit_file", {"file_path": "/a.py"}, messages)
        tool_result = ToolMessage(content="OK", tool_call_id="call-1")
        handler = Mock(return_value=tool_result)

        result = mw.wrap_tool_call(request, handler)

        handler.assert_called_once_with(request)
        assert isinstance(result, ToolMessage)
        assert "NOTE" not in str(result.content)

    def test_warning_at_threshold(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=4, hard_threshold=8)
        # 3 prior edits + 1 current = 4, hits threshold
        messages = _make_edit_calls("/a.py", 3)
        request = _make_tool_call_request("edit_file", {"file_path": "/a.py"}, messages)
        tool_result = ToolMessage(content="OK", tool_call_id="call-1")
        handler = Mock(return_value=tool_result)

        result = mw.wrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert "**NOTE**" in str(result.content)
        assert "/a.py" in str(result.content)

    def test_warning_only_once(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=4, hard_threshold=8)
        # 4 prior edits + soft warning already in messages
        messages = [
            *_make_edit_calls("/a.py", 4),
            ToolMessage(
                content="OK" + LOOP_WARNING_SOFT.format(file_path="/a.py", count=4),
                tool_call_id="prev",
            ),
        ]
        request = _make_tool_call_request("edit_file", {"file_path": "/a.py"}, messages)
        tool_result = ToolMessage(content="OK", tool_call_id="call-1")
        handler = Mock(return_value=tool_result)

        result = mw.wrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        # Should NOT have a new warning appended
        assert str(result.content) == "OK"

    def test_non_edit_tool_passes_through(self) -> None:
        mw = LoopDetectionMiddleware()
        request = _make_tool_call_request("read_file", {"file_path": "/a.py"}, [])
        tool_result = ToolMessage(content="file contents", tool_call_id="call-1")
        handler = Mock(return_value=tool_result)

        result = mw.wrap_tool_call(request, handler)

        assert str(result.content) == "file contents"


# ---------------------------------------------------------------------------
# Soft warning (async)
# ---------------------------------------------------------------------------


class TestSoftWarningAsync:
    @pytest.mark.asyncio
    async def test_async_warning_at_threshold(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=4, hard_threshold=8)
        messages = _make_edit_calls("/a.py", 3)
        request = _make_tool_call_request("edit_file", {"file_path": "/a.py"}, messages)
        tool_result = ToolMessage(content="OK", tool_call_id="call-1")

        async def async_handler(_req: object) -> ToolMessage:  # noqa: RUF029
            return tool_result

        result = await mw.awrap_tool_call(request, async_handler)

        assert isinstance(result, ToolMessage)
        assert "**NOTE**" in str(result.content)


# ---------------------------------------------------------------------------
# Hard warning (after_model)
# ---------------------------------------------------------------------------


class TestHardWarning:
    def test_no_warning_below_threshold(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=4, hard_threshold=8)
        # 7 edits, below hard threshold of 8
        messages = [
            *_make_edit_calls("/a.py", 7),
            ToolMessage(content="OK", tool_call_id="last"),
        ]
        state = cast("Any", {"messages": messages})

        result = mw.after_model(state, Mock())
        assert result is None

    def test_warning_at_threshold(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=4, hard_threshold=8)
        # 8 edits = hard threshold
        messages = [
            *_make_edit_calls("/a.py", 8),
            ToolMessage(content="OK", tool_call_id="last"),
        ]
        state = cast("Any", {"messages": messages})

        result = mw.after_model(state, Mock())

        assert result is not None
        assert result["jump_to"] == "model"
        injected = result["messages"][0]
        assert isinstance(injected, HumanMessage)
        assert "/a.py" in injected.content
        assert "8 times" in injected.content

    def test_skipped_when_tool_calls_pending(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=4, hard_threshold=8)
        # 8 edits but last message has pending tool calls
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "edit_file", "args": {"file_path": "/a.py"}, "id": "pending"}
            ],
        )
        messages = [*_make_edit_calls("/a.py", 7), ai_msg]
        state = cast("Any", {"messages": messages})

        result = mw.after_model(state, Mock())
        assert result is None

    def test_warning_only_once(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=4, hard_threshold=8)
        # 8 edits + hard warning already injected
        messages = [
            *_make_edit_calls("/a.py", 8),
            HumanMessage(content=LOOP_WARNING_HARD.format(file_path="/a.py", count=8)),
            ToolMessage(content="OK", tool_call_id="after-hard"),
        ]
        state = cast("Any", {"messages": messages})

        result = mw.after_model(state, Mock())
        assert result is None

    @pytest.mark.asyncio
    async def test_async_warning_at_threshold(self) -> None:
        mw = LoopDetectionMiddleware(soft_threshold=4, hard_threshold=8)
        messages = [
            *_make_edit_calls("/a.py", 8),
            ToolMessage(content="OK", tool_call_id="last"),
        ]
        state = cast("Any", {"messages": messages})

        result = await mw.aafter_model(state, Mock())

        assert result is not None
        assert result["jump_to"] == "model"


# ---------------------------------------------------------------------------
# Concurrent safety (same instance, different states)
# ---------------------------------------------------------------------------


class TestConcurrentSafety:
    def test_independent_states(self) -> None:
        """Same middleware instance, two different state dicts → independent."""
        mw = LoopDetectionMiddleware(soft_threshold=2, hard_threshold=4)

        # State A: 1 edit to /a.py (+ 1 current = 2, hits soft)
        messages_a = _make_edit_calls("/a.py", 1)
        request_a = _make_tool_call_request(
            "edit_file", {"file_path": "/a.py"}, messages_a
        )
        tool_result_a = ToolMessage(content="OK-A", tool_call_id="call-a")
        handler_a = Mock(return_value=tool_result_a)

        # State B: 0 edits to /a.py (+ 1 current = 1, no warning)
        request_b = _make_tool_call_request("edit_file", {"file_path": "/a.py"}, [])
        tool_result_b = ToolMessage(content="OK-B", tool_call_id="call-b")
        handler_b = Mock(return_value=tool_result_b)

        result_a = mw.wrap_tool_call(request_a, handler_a)
        result_b = mw.wrap_tool_call(request_b, handler_b)

        # A should have warning, B should not
        assert "**NOTE**" in str(result_a.content)
        assert "**NOTE**" not in str(result_b.content)

    def test_hard_warning_independent_states(self) -> None:
        """after_model with different states should be independent."""
        mw = LoopDetectionMiddleware(soft_threshold=2, hard_threshold=4)

        # State A: 4 edits → hard warning
        state_a = cast(
            "Any",
            {
                "messages": [
                    *_make_edit_calls("/a.py", 4),
                    ToolMessage(content="OK", tool_call_id="last"),
                ]
            },
        )
        # State B: 3 edits → no hard warning
        state_b = cast(
            "Any",
            {
                "messages": [
                    *_make_edit_calls("/a.py", 3),
                    ToolMessage(content="OK", tool_call_id="last"),
                ]
            },
        )

        result_a = mw.after_model(state_a, Mock())
        result_b = mw.after_model(state_b, Mock())

        assert result_a is not None
        assert result_a["jump_to"] == "model"
        assert result_b is None
