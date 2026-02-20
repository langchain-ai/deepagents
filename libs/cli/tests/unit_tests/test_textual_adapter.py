"""Unit tests for textual_adapter functions."""

from asyncio import Future
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from deepagents_cli.textual_adapter import (
    TextualUIAdapter,
    _build_interrupted_ai_message,
    _build_stream_config,
    _is_summarization_chunk,
)


async def _mock_mount(widget: object) -> None:
    """Mock mount function for tests."""


def _mock_approval() -> Future[object]:
    """Mock approval function for tests."""
    future: Future[object] = Future()
    return future


def _noop_status(_: str) -> None:
    """No-op status callback for tests."""


class TestTextualUIAdapterInit:
    """Tests for `TextualUIAdapter` initialization."""

    def test_set_spinner_callback_stored(self) -> None:
        """Verify `set_spinner` callback is properly stored."""

        async def mock_spinner(status: str | None) -> None:
            pass

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=mock_spinner,
        )
        assert adapter._set_spinner is mock_spinner

    def test_set_spinner_defaults_to_none(self) -> None:
        """Verify `set_spinner` is optional and defaults to `None`."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._set_spinner is None

    def test_current_tool_messages_initialized_empty(self) -> None:
        """Verify `_current_tool_messages` is initialized as empty dict."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._current_tool_messages == {}

    def test_token_tracker_initialized_none(self) -> None:
        """Verify `_token_tracker` is initialized as `None`."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._token_tracker is None

    def test_set_token_tracker(self) -> None:
        """Verify `set_token_tracker` stores the tracker."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        mock_tracker = object()
        adapter.set_token_tracker(mock_tracker)
        assert adapter._token_tracker is mock_tracker

    def test_finalize_pending_tools_with_error_marks_and_clears(self) -> None:
        """Pending tool widgets should be marked error and then cleared."""
        set_active = MagicMock()
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_active_message=set_active,
        )

        tool_1 = MagicMock()
        tool_2 = MagicMock()
        adapter._current_tool_messages = {"a": tool_1, "b": tool_2}

        adapter.finalize_pending_tools_with_error("Agent error: boom")

        tool_1.set_error.assert_called_once_with("Agent error: boom")
        tool_2.set_error.assert_called_once_with("Agent error: boom")
        assert adapter._current_tool_messages == {}
        set_active.assert_called_once_with(None)


class TestBuildStreamConfig:
    """Tests for `_build_stream_config` metadata construction."""

    def test_assistant_fields_present(self) -> None:
        """Assistant-specific metadata should be present when `assistant_id` is set."""
        config = _build_stream_config("t-456", assistant_id="my-agent")
        assert config["metadata"]["assistant_id"] == "my-agent"
        assert config["metadata"]["agent_name"] == "my-agent"
        assert "updated_at" in config["metadata"]

    def test_updated_at_is_valid_iso_timestamp(self) -> None:
        """`updated_at` should be a valid timezone-aware ISO 8601 timestamp."""
        config = _build_stream_config("t-456", assistant_id="my-agent")
        raw = config["metadata"]["updated_at"]
        assert isinstance(raw, str)
        parsed = datetime.fromisoformat(raw)
        assert parsed.tzinfo is not None

    def test_no_assistant_fields_when_none(self) -> None:
        """Assistant-specific fields should be absent when `assistant_id` is `None`."""
        config = _build_stream_config("t-789", assistant_id=None)
        assert config["metadata"] == {}

    def test_no_assistant_fields_when_empty_string(self) -> None:
        """Empty-string `assistant_id` should be treated as absent."""
        config = _build_stream_config("t-000", assistant_id="")
        assert config["metadata"] == {}

    def test_configurable_thread_id(self) -> None:
        """`configurable.thread_id` should match the provided thread ID."""
        config = _build_stream_config("t-abc", assistant_id=None)
        assert config["configurable"]["thread_id"] == "t-abc"


class TestIsSummarizationChunk:
    """Tests for `_is_summarization_chunk` detection."""

    def test_returns_true_for_summarization_source(self) -> None:
        """Should return `True` when `lc_source` is `'summarization'`."""
        metadata = {"lc_source": "summarization"}
        assert _is_summarization_chunk(metadata) is True

    def test_returns_false_for_none_metadata(self) -> None:
        """Should return `False` when `metadata` is `None`."""
        assert _is_summarization_chunk(None) is False
        assert _is_summarization_chunk({}) is False

    def test_returns_false_for_none_lc_source(self) -> None:
        """Should return `False` when `lc_source` is not `'summarization'`."""
        metadata_none = {"lc_source": None}
        assert _is_summarization_chunk(metadata_none) is False

        metadata_other = {"lc_source": "other"}
        assert _is_summarization_chunk(metadata_other) is False

        metadata_missing = {"other_key": "value"}
        assert _is_summarization_chunk(metadata_missing) is False


# ---------------------------------------------------------------------------
# Helpers for dict-iteration safety tests
# ---------------------------------------------------------------------------


def _make_tool_widget(name: str = "tool", args: dict | None = None) -> MagicMock:
    """Create a MagicMock that mimics a ToolCallMessage widget."""
    widget = MagicMock()
    widget._tool_name = name
    widget._args = args or {}
    return widget


class _MutatingItemsDict(dict):  # noqa: FURB189  # must subclass dict to override C-level iteration
    """Dict whose `.items()` deletes another key mid-iteration.

    This deterministically reproduces the `RuntimeError: dictionary
    changed size during iteration` that occurs when async tool-result
    callbacks mutate `_current_tool_messages` while the HITL approval
    loop is iterating over it.

    We intentionally subclass `dict` (not `UserDict`) because we
    need to override the C-level iteration that triggers the error.
    """

    def items(self) -> Generator[tuple[str, Any], None, None]:  # type: ignore[override]
        """Yield items while mutating the dict mid-iteration."""
        it = iter(dict.items(self))
        first = next(it)
        # Remove a *different* key while iteration is in progress.
        remaining = [k for k in self if k != first[0]]
        if remaining:
            del self[remaining[0]]
        yield first
        yield from it


class _MutatingValuesDict(dict):  # noqa: FURB189  # must subclass dict to override C-level iteration
    """Dict whose `.values()` deletes a key mid-iteration.

    We intentionally subclass `dict` (not `UserDict`) because we
    need to override the C-level iteration that triggers the error.
    """

    def values(self) -> Generator[Any, None, None]:  # type: ignore[override]
        """Yield values while mutating the dict mid-iteration."""
        it = iter(dict.values(self))
        first = next(it)
        # Remove the first key to trigger size-change error.
        first_key = next(iter(self))
        del self[first_key]
        yield first
        yield from it


class TestDictIterationSafety:
    """Regression tests for #956.

    Parallel tool calls can modify `adapter._current_tool_messages`
    while another coroutine iterates over it, raising
    `RuntimeError: dictionary changed size during iteration`.

    The fix wraps every iteration with `list()` so a snapshot is
    taken before the loop body runs.  These tests prove the fix is
    necessary and sufficient.
    """

    # -- Test A: bare iteration over a mutating dict raises ----

    def test_items_iteration_fails_without_list(self) -> None:
        """Iterating .items() on a concurrently-mutated dict raises."""
        d = _MutatingItemsDict(
            {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(3)}
        )
        with pytest.raises(RuntimeError, match="changed size"):
            for _ in d.items():
                pass

    def test_values_iteration_fails_without_list(self) -> None:
        """Iterating .values() on a concurrently-mutated dict raises."""
        d = _MutatingValuesDict(
            {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(3)}
        )
        with pytest.raises(RuntimeError, match="changed size"):
            for _ in d.values():
                pass

    # -- Test B: list() snapshot protects iteration ----

    def test_items_iteration_safe_with_list(self) -> None:
        """`list(d.items())` snapshots before mutation can occur."""
        d: dict = {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(5)}
        collected = []
        for key, _val in list(d.items()):
            collected.append(key)
            d.pop(key, None)  # mutate during loop body
        assert len(collected) == 5
        assert len(d) == 0

    def test_values_iteration_safe_with_list(self) -> None:
        """`list(d.values())` snapshots before mutation."""
        d: dict = {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(5)}
        collected = []
        keys = list(d.keys())
        for val in list(d.values()):
            collected.append(val)
            if keys:
                d.pop(keys.pop(0), None)
        assert len(collected) == 5

    # -- Test C: _build_interrupted_ai_message uses list() ----

    def test_build_interrupted_ai_message_safe(self) -> None:
        """_build_interrupted_ai_message correctly builds an AIMessage.

        Verifies the function reconstructs tool calls and content from
        the provided widget dict. The `list()` snapshot inside the
        production code protects against external async mutation at
        `await` boundaries, which cannot be deterministically simulated
        in a synchronous unit test.
        """
        widgets = {
            f"id_{i}": _make_tool_widget(f"tool_{i}", {"k": i}) for i in range(4)
        }
        pending_text: dict[tuple, str] = {(): "hello"}
        result = _build_interrupted_ai_message(pending_text, widgets)
        assert result is not None
        assert result.content == "hello"
        assert len(result.tool_calls) == 4
        names = {tc["name"] for tc in result.tool_calls}
        assert names == {"tool_0", "tool_1", "tool_2", "tool_3"}

    def test_build_interrupted_ai_message_empty(self) -> None:
        """Returns None when there is no text and no tool calls."""
        result = _build_interrupted_ai_message({}, {})
        assert result is None
