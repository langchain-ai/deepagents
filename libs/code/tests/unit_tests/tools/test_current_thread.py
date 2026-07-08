"""Tests for the `get_current_thread_id` tool."""

from __future__ import annotations

from deepagents_code.tools import get_current_thread_id


def test_get_current_thread_id_returns_config_thread() -> None:
    """Tool should read the injected LangGraph thread ID."""
    result = get_current_thread_id.invoke(
        {},
        config={"configurable": {"thread_id": "thread-123"}},
    )

    assert result == "thread-123"


def test_get_current_thread_id_has_no_model_visible_args() -> None:
    """Runtime config injection should keep the schema argument-free."""
    assert get_current_thread_id.args == {}


def test_get_current_thread_id_handles_missing_thread() -> None:
    """Tool should return a clear message when no thread ID is present."""
    result = get_current_thread_id.invoke({}, config={"configurable": {}})

    assert result == "No current thread ID is available."
