"""Unit tests for message widget timestamp tooltips."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from textual.app import App, ComposeResult

from deepagents_code.widgets.message_store import MessageData, MessageStore, MessageType
from deepagents_code.widgets.messages import UserMessage, _apply_timestamp_tooltip

if TYPE_CHECKING:
    from collections.abc import Iterator

    from textual.widget import Widget

# Fixed epoch -> "Jan 1, 12:00:05 PM" UTC, used to pin the format contract.
_FIXED_TIMESTAMP = 1_704_110_405.0
_FIXED_LABEL = "Jan 1, 12:00:05 PM"


@pytest.fixture
def utc_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run timestamp assertions in UTC."""
    previous = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "UTC")
    if hasattr(time, "tzset"):
        time.tzset()
    yield
    if previous is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", previous)
    if hasattr(time, "tzset"):
        time.tzset()


class _Harness(App[None]):
    """Mounts the given widgets, optionally backed by a message store."""

    def __init__(
        self,
        widgets: list[Widget],
        messages: list[MessageData],
        *,
        with_store: bool = True,
    ) -> None:
        super().__init__()
        self.widgets = widgets
        if with_store:
            self._message_store = MessageStore()
            for message in messages:
                self._message_store.append(message)

    def compose(self) -> ComposeResult:
        yield from self.widgets


def _all_message_data() -> list[MessageData]:
    """One `MessageData` per `MessageType`, sharing a fixed timestamp."""
    return [
        MessageData(
            type=MessageType.USER, content="hi", id="m-user", timestamp=_FIXED_TIMESTAMP
        ),
        MessageData(
            type=MessageType.ASSISTANT,
            content="hello",
            id="m-assistant",
            timestamp=_FIXED_TIMESTAMP,
        ),
        MessageData(
            type=MessageType.TOOL,
            content="",
            id="m-tool",
            tool_name="ls",
            timestamp=_FIXED_TIMESTAMP,
        ),
        MessageData(
            type=MessageType.SKILL,
            content="",
            id="m-skill",
            skill_name="demo",
            timestamp=_FIXED_TIMESTAMP,
        ),
        MessageData(
            type=MessageType.ERROR,
            content="boom",
            id="m-error",
            timestamp=_FIXED_TIMESTAMP,
        ),
        MessageData(
            type=MessageType.APP, content="note", id="m-app", timestamp=_FIXED_TIMESTAMP
        ),
        MessageData(
            type=MessageType.SUMMARIZATION,
            content="summary",
            id="m-summary",
            timestamp=_FIXED_TIMESTAMP,
        ),
        MessageData(
            type=MessageType.DIFF,
            content="@@ -1 +1 @@",
            id="m-diff",
            diff_file_path="a.py",
            timestamp=_FIXED_TIMESTAMP,
        ),
    ]


@pytest.mark.usefixtures("utc_timezone")
async def test_hydrated_widgets_get_timestamp_tooltip() -> None:
    """Every message widget type applies the tooltip on mount via the store.

    Exercises the history-hydration contract: widgets are recreated from
    persisted `MessageData` via `to_widget()`, mounted, and must each pick up
    their creation timestamp through their own `on_mount`.
    """
    data = _all_message_data()
    widgets = [d.to_widget() for d in data]
    app = _Harness(widgets, data)

    async with app.run_test() as pilot:
        await pilot.pause()

        for widget in widgets:
            assert widget.tooltip == _FIXED_LABEL, widget


@pytest.mark.usefixtures("utc_timezone")
async def test_apply_timestamp_tooltip_formats_fixed_timestamp() -> None:
    """The format string pins the 12-hour, locale-stable label contract."""
    data = MessageData(
        type=MessageType.USER,
        content="hello",
        id="msg-fixed",
        timestamp=_FIXED_TIMESTAMP,
    )
    app = _Harness([UserMessage("hello", id=data.id)], [data])

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.widgets[0].tooltip == _FIXED_LABEL


async def test_missing_store_entry_leaves_tooltip_none() -> None:
    """A widget whose id is not registered in the store gets no tooltip."""
    app = _Harness([UserMessage("hello", id="msg-missing")], [])

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.widgets[0].tooltip is None


async def test_widget_without_id_leaves_tooltip_none() -> None:
    """A widget without a DOM id cannot be looked up, so no tooltip is set."""
    app = _Harness([UserMessage("hello")], [])

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.widgets[0].tooltip is None


async def test_missing_message_store_leaves_tooltip_none() -> None:
    """Hosts without a `_message_store` attribute no-op without raising."""
    data = MessageData(
        type=MessageType.USER,
        content="hello",
        id="msg-no-store",
        timestamp=_FIXED_TIMESTAMP,
    )
    app = _Harness([UserMessage("hello", id=data.id)], [data], with_store=False)

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.widgets[0].tooltip is None


def test_unmounted_widget_no_ops() -> None:
    """Calling the helper on a widget with no active app returns silently."""
    widget = UserMessage("hello", id="msg-unmounted")

    _apply_timestamp_tooltip(widget)

    assert widget.tooltip is None


@pytest.mark.usefixtures("utc_timezone")
async def test_invalid_timestamp_leaves_tooltip_none() -> None:
    """A malformed stored timestamp degrades to no tooltip instead of crashing."""
    data = MessageData(
        type=MessageType.USER,
        content="hello",
        id="msg-bad-ts",
        timestamp=_FIXED_TIMESTAMP,
    )
    widget = UserMessage("hello", id=data.id)
    app = _Harness([widget], [data])

    async with app.run_test() as pilot:
        await pilot.pause()
        widget.tooltip = None
        # Corrupt the stored timestamp to an out-of-range value.
        stored = app._message_store.get_message(data.id)
        assert stored is not None
        stored.timestamp = float("inf")
        _apply_timestamp_tooltip(widget)

        assert widget.tooltip is None
