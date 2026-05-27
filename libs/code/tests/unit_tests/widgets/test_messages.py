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
    def __init__(self, widget: UserMessage, messages: list[MessageData]) -> None:
        super().__init__()
        self.widget = widget
        self._message_store = MessageStore()
        for message in messages:
            self._message_store.append(message)

    def compose(self) -> ComposeResult:
        yield self.widget


def _timestamp_label(timestamp: float) -> str:
    dt = datetime.fromtimestamp(timestamp, tz=UTC).astimezone()
    return f"{dt:%b} {dt.day}, {dt.hour % 12 or 12}:{dt:%M:%S} {dt:%p}"


@pytest.mark.usefixtures("utc_timezone")
async def test_user_message_mount_applies_timestamp_tooltip() -> None:
    timestamp = 1_704_110_405.0
    data = MessageData(
        type=MessageType.USER,
        content="hello",
        id="msg-known",
        timestamp=timestamp,
    )
    app = _Harness(UserMessage("hello", id=data.id), [data])

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.widget.tooltip == _timestamp_label(timestamp)


async def test_user_message_missing_store_entry_leaves_tooltip_none() -> None:
    app = _Harness(UserMessage("hello", id="msg-missing"), [])

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.widget.tooltip is None


@pytest.mark.usefixtures("utc_timezone")
async def test_apply_timestamp_tooltip_formats_fixed_timestamp() -> None:
    timestamp = 1_704_110_405.0
    data = MessageData(
        type=MessageType.USER,
        content="hello",
        id="msg-fixed",
        timestamp=timestamp,
    )
    app = _Harness(UserMessage("hello", id=data.id), [data])

    async with app.run_test() as pilot:
        await pilot.pause()
        app.widget.tooltip = None
        _apply_timestamp_tooltip(app.widget)

        assert app.widget.tooltip == "Jan 1, 12:00:05 PM"
