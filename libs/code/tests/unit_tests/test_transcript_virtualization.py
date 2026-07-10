"""Focused TUI tests for transcript virtualization scroll hydration.

These bind the behavior that history hydration is driven by real changes to the
chat's vertical scroll offset (`_ChatScroll.watch_scroll_y` -> the
`_ChatScroll.Scrolled` message -> `DeepAgentsApp.on_chat_scrolled`), rather than
the scrollbar `ScrollUp`/`ScrollDown` messages the feature originally relied on.
Those scrollbar messages never fire for wheel/trackpad/keyboard scrolling and are
consumed by the container before reaching the app, so hydration never ran for the
common case of scrolling with a trackpad.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static

from deepagents_code.app import DeepAgentsApp, _ChatScroll
from deepagents_code.tui.widgets.messages import UserMessage

if TYPE_CHECKING:
    import pytest


class _ScrollProbeApp(App[None]):
    """Minimal app that counts `_ChatScroll.Scrolled` notifications."""

    CSS = """
    #chat {
        height: 6;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.scroll_notifications = 0

    def compose(self) -> ComposeResult:
        """Compose an overflowing transcript inside a `_ChatScroll`."""
        with _ChatScroll(id="chat"):
            yield Static("\n".join(f"line {index}" for index in range(60)))

    def on_chat_scrolled(self, _event: _ChatScroll.Scrolled) -> None:
        self.scroll_notifications += 1


class TestChatScrollNotifies:
    """`_ChatScroll` must announce every scroll offset change to the app."""

    async def test_wheel_scroll_notifies_app(self) -> None:
        """Wheel/trackpad scrolling (MouseScroll events) reaches `on_chat_scrolled`."""
        app = _ScrollProbeApp()
        async with app.run_test(size=(40, 6)) as pilot:
            chat = app.query_one("#chat", _ChatScroll)
            chat.scroll_end(animate=False)
            await pilot.pause()
            assert chat.max_scroll_y > 0
            app.scroll_notifications = 0

            # A trackpad/wheel scroll is delivered as a MouseScrollUp event,
            # never as a scrollbar ScrollUp message.
            chat.post_message(
                events.MouseScrollUp(
                    widget=chat,
                    x=1,
                    y=1,
                    delta_x=0,
                    delta_y=-1,
                    button=0,
                    shift=False,
                    meta=False,
                    ctrl=False,
                )
            )
            await pilot.pause()
            await pilot.pause()

            assert chat.scroll_y < chat.max_scroll_y
            assert app.scroll_notifications > 0

    async def test_scrollbar_track_scroll_notifies_app(self) -> None:
        """Scrollbar-track paging also flows through the scroll-offset watcher."""
        app = _ScrollProbeApp()
        async with app.run_test(size=(40, 6)) as pilot:
            chat = app.query_one("#chat", _ChatScroll)
            await pilot.pause()
            app.scroll_notifications = 0

            chat._vertical_scrollbar.action_scroll_down()
            await pilot.pause()
            await pilot.pause()

            assert app.scroll_notifications > 0

    async def test_unchanged_offset_does_not_notify(self) -> None:
        """Re-setting the same offset must not churn the app with notifications."""
        app = _ScrollProbeApp()
        async with app.run_test(size=(40, 6)) as pilot:
            chat = app.query_one("#chat", _ChatScroll)
            await pilot.pause()
            app.scroll_notifications = 0

            chat.scroll_y = chat.scroll_y
            await pilot.pause()

            assert app.scroll_notifications == 0


async def _mount_user_messages(app: DeepAgentsApp, count: int) -> None:
    """Mount `count` `UserMessage` rows through the real mount path."""
    for index in range(count):
        await app._mount_message(UserMessage(f"m{index}", id=f"m{index}"))


class TestScrollDrivenHydration:
    """Scrolling into a spacer must hydrate the adjacent archived history."""

    async def test_scroll_up_hydrates_archived_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scrolling up toward the top spacer remounts older messages."""
        app = DeepAgentsApp()
        async with app.run_test(size=(80, 10)) as pilot:
            await pilot.pause()
            await _mount_user_messages(app, 20)
            await pilot.pause()

            # Shrink the window and prune the oldest rows so history is archived
            # above the mounted tail (the state after a long transcript grows).
            monkeypatch.setattr(app._message_store, "WINDOW_SIZE", 3)
            await app._prune_old_messages()
            await pilot.pause()

            start_before, _end_before = app._message_store.get_visible_range()
            assert app._message_store.has_messages_above
            assert start_before > 0

            chat = app.query_one("#chat", _ChatScroll)
            chat.scroll_end(animate=False)
            await pilot.pause()
            chat.scroll_to(y=0, animate=False)
            for _ in range(6):
                await pilot.pause()

            start_after, _end_after = app._message_store.get_visible_range()
            assert start_after < start_before

    async def test_scroll_down_hydrates_tail_below(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scrolling down toward the bottom spacer remounts newer messages."""
        app = DeepAgentsApp()
        async with app.run_test(size=(80, 10)) as pilot:
            await pilot.pause()
            await _mount_user_messages(app, 20)
            await pilot.pause()

            # Archive the newest rows below the window (the state after the user
            # has scrolled up and older history was mounted in their place).
            monkeypatch.setattr(app._message_store, "WINDOW_SIZE", 3)
            messages = app.query_one("#messages", Container)
            await app._prune_messages_below_window(messages)
            await pilot.pause()

            _start_before, end_before = app._message_store.get_visible_range()
            assert app._message_store.has_messages_below

            chat = app.query_one("#chat", _ChatScroll)
            chat.scroll_to(y=0, animate=False)
            await pilot.pause()
            chat.scroll_end(animate=False)
            for _ in range(6):
                await pilot.pause()

            _start_after, end_after = app._message_store.get_visible_range()
            assert end_after > end_before
