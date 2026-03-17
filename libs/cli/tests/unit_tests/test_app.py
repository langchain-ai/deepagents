"""Unit tests for DeepAgentsApp."""

from __future__ import annotations

import asyncio
import inspect
import io
import os
import signal
import time
import webbrowser
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import AsyncMock, MagicMock, call, patch

if TYPE_CHECKING:
    from deepagents_cli.sessions import ThreadInfo

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Input, Static

from deepagents_cli.app import (
    _ITERM_CURSOR_GUIDE_OFF,
    _ITERM_CURSOR_GUIDE_ON,
    _TYPING_IDLE_THRESHOLD_SECONDS,
    DeepAgentsApp,
    QueuedMessage,
    TextualSessionState,
    _write_iterm_escape,
)
from deepagents_cli.widgets.chat_input import ChatInput
from deepagents_cli.widgets.messages import (
    AppMessage,
    ErrorMessage,
    QueuedUserMessage,
    UserMessage,
)


class TestInitialPromptOnMount:
    """Test that -m initial prompt is submitted on mount."""

    async def test_initial_prompt_triggers_handle_user_message(self) -> None:
        """When initial_prompt is set, the prompt should be auto-submitted."""
        mock_agent = MagicMock()
        app = DeepAgentsApp(
            agent=mock_agent,
            thread_id="new-thread-123",
            initial_prompt="hello world",
        )
        submitted: list[str] = []

        # Must be async to match _handle_user_message's signature
        async def capture(msg: str) -> None:  # noqa: RUF029
            submitted.append(msg)

        app._handle_user_message = capture  # type: ignore[assignment]

        async with app.run_test() as pilot:
            # Give call_after_refresh time to fire
            await pilot.pause()
            await pilot.pause()

        assert submitted == ["hello world"]


class TestAppCSSValidation:
    """Test that app CSS is valid and doesn't cause runtime errors."""

    async def test_app_css_validates_on_mount(self) -> None:
        """App should mount without CSS validation errors.

        This test catches invalid CSS properties like 'overflow: visible'
        which are only validated at runtime when styles are applied.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            # Give the app time to render and apply CSS
            await pilot.pause()
            # If we get here without exception, CSS is valid
            assert app.is_running


class TestThreadCachePrewarm:
    """Tests for startup thread-cache prewarming."""

    async def test_prewarm_uses_current_thread_limit(self) -> None:
        """Prewarm helper should pass the resolved thread limit through."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")

        with (
            patch("deepagents_cli.sessions.get_thread_limit", return_value=7),
            patch(
                "deepagents_cli.sessions.prewarm_thread_message_counts",
                new_callable=AsyncMock,
            ) as mock_prewarm,
        ):
            await app._prewarm_threads_cache()

        mock_prewarm.assert_awaited_once_with(limit=7)

    async def test_show_thread_selector_uses_cached_rows(self) -> None:
        """Thread selector should receive prefetched rows when available."""
        cached_threads = [
            {
                "thread_id": "thread-abc",
                "agent_name": "agent1",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "message_count": 2,
            }
        ]
        app = DeepAgentsApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            with (
                patch("deepagents_cli.sessions.get_thread_limit", return_value=9),
                patch(
                    "deepagents_cli.sessions.get_cached_threads",
                    return_value=cached_threads,
                ),
                patch("deepagents_cli.app.ThreadSelectorScreen") as mock_screen_cls,
                patch.object(app, "push_screen") as mock_push_screen,
            ):
                mock_screen = MagicMock()
                mock_screen_cls.return_value = mock_screen
                await app._show_thread_selector()

                assert app._session_state is not None
                mock_screen_cls.assert_called_once_with(
                    current_thread=app._session_state.thread_id,
                    thread_limit=9,
                    initial_threads=cached_threads,
                )
                mock_push_screen.assert_called_once()


class TestAppBindings:
    """Test app keybindings."""

    def test_ctrl_c_binding_has_priority(self) -> None:
        """Ctrl+C should be priority-bound so focused modal inputs don't swallow it."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        ctrl_c = bindings_by_key.get("ctrl+c")

        assert ctrl_c is not None
        assert ctrl_c.action == "quit_or_interrupt"
        assert ctrl_c.priority is True

    def test_toggle_tool_output_has_ctrl_e_binding(self) -> None:
        """Ctrl+E should be bound to toggle_tool_output with priority."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        ctrl_e = bindings_by_key.get("ctrl+e")

        assert ctrl_e is not None
        assert ctrl_e.action == "toggle_tool_output"
        assert ctrl_e.priority is True

    def test_ctrl_o_not_bound_to_toggle_tool_output(self) -> None:
        """Ctrl+O should not exist (replaced by Ctrl+E)."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        assert "ctrl+o" not in bindings_by_key


class TestITerm2CursorGuide:
    """Test iTerm2 cursor guide handling."""

    def test_escape_sequences_are_valid(self) -> None:
        """Escape sequences should be properly formatted OSC 1337 commands.

        Format: OSC (ESC ]) + "1337;" + command + ST (ESC backslash)
        """
        assert _ITERM_CURSOR_GUIDE_OFF.startswith("\x1b]1337;")
        assert _ITERM_CURSOR_GUIDE_OFF.endswith("\x1b\\")
        assert "HighlightCursorLine=no" in _ITERM_CURSOR_GUIDE_OFF

        assert _ITERM_CURSOR_GUIDE_ON.startswith("\x1b]1337;")
        assert _ITERM_CURSOR_GUIDE_ON.endswith("\x1b\\")
        assert "HighlightCursorLine=yes" in _ITERM_CURSOR_GUIDE_ON

    def test_write_iterm_escape_does_nothing_when_not_iterm(self) -> None:
        """_write_iterm_escape should no-op when _IS_ITERM is False."""
        mock_stderr = MagicMock()
        with (
            patch("deepagents_cli.app._IS_ITERM", False),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
            mock_stderr.write.assert_not_called()

    def test_write_iterm_escape_writes_sequence_when_iterm(self) -> None:
        """_write_iterm_escape should write sequence when in iTerm2."""
        mock_stderr = io.StringIO()
        with (
            patch("deepagents_cli.app._IS_ITERM", True),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
            assert mock_stderr.getvalue() == _ITERM_CURSOR_GUIDE_ON

    def test_write_iterm_escape_handles_oserror_gracefully(self) -> None:
        """_write_iterm_escape should not raise on OSError."""
        mock_stderr = MagicMock()
        mock_stderr.write.side_effect = OSError("Broken pipe")
        with (
            patch("deepagents_cli.app._IS_ITERM", True),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)

    def test_write_iterm_escape_handles_none_stderr(self) -> None:
        """_write_iterm_escape should handle None __stderr__ gracefully."""
        with (
            patch("deepagents_cli.app._IS_ITERM", True),
            patch("sys.__stderr__", None),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)


class TestITerm2Detection:
    """Test iTerm2 detection logic."""

    def test_detection_requires_tty(self) -> None:
        """_IS_ITERM should check that stderr is a TTY.

        Detection happens at module load, so we test the logic pattern directly.
        """
        with (
            patch.dict(os.environ, {"LC_TERMINAL": "iTerm2"}, clear=False),
            patch("os.isatty", return_value=False),
        ):
            result = (
                (
                    os.environ.get("LC_TERMINAL", "") == "iTerm2"
                    or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
                )
                and hasattr(os, "isatty")
                and os.isatty(2)
            )
            assert result is False

    def test_detection_via_lc_terminal(self) -> None:
        """Detection should match LC_TERMINAL=iTerm2."""
        with (
            patch.dict(
                os.environ, {"LC_TERMINAL": "iTerm2", "TERM_PROGRAM": ""}, clear=False
            ),
            patch("os.isatty", return_value=True),
        ):
            result = (
                (
                    os.environ.get("LC_TERMINAL", "") == "iTerm2"
                    or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
                )
                and hasattr(os, "isatty")
                and os.isatty(2)
            )
            assert result is True

    def test_detection_via_term_program(self) -> None:
        """Detection should match TERM_PROGRAM=iTerm.app."""
        env = {"LC_TERMINAL": "", "TERM_PROGRAM": "iTerm.app"}
        with (
            patch.dict(os.environ, env, clear=False),
            patch("os.isatty", return_value=True),
        ):
            result = (
                (
                    os.environ.get("LC_TERMINAL", "") == "iTerm2"
                    or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
                )
                and hasattr(os, "isatty")
                and os.isatty(2)
            )
            assert result is True


class TestModalScreenEscapeDismissal:
    """Test that escape key dismisses modal screens."""

    @staticmethod
    async def test_escape_dismisses_modal_screen() -> None:
        """Escape should dismiss any active ModalScreen.

        The app's action_interrupt binding intercepts escape with priority=True.
        When a modal screen is active, it should dismiss the modal rather than
        performing the default interrupt behavior.
        """

        class SimpleModal(ModalScreen[str | None]):
            """A simple test modal."""

            BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

            def compose(self) -> ComposeResult:
                yield Static("Test Modal")

            def action_cancel(self) -> None:
                self.dismiss(None)

        class TestApp(App[None]):
            """Test app with escape -> action_interrupt binding."""

            BINDINGS: ClassVar[list[BindingType]] = [
                Binding("escape", "interrupt", "Interrupt", priority=True)
            ]

            def __init__(self) -> None:
                super().__init__()
                self.modal_dismissed = False
                self.interrupt_called = False

            def compose(self) -> ComposeResult:
                yield Container()

            def action_interrupt(self) -> None:
                if isinstance(self.screen, ModalScreen):
                    self.screen.dismiss(None)
                    return
                self.interrupt_called = True

            def show_modal(self) -> None:
                def on_dismiss(_result: str | None) -> None:
                    self.modal_dismissed = True

                self.push_screen(SimpleModal(), on_dismiss)

        app = TestApp()
        async with app.run_test() as pilot:
            app.show_modal()
            await pilot.pause()

            # Escape should dismiss the modal, not call interrupt
            await pilot.press("escape")
            await pilot.pause()

            assert app.modal_dismissed is True
            assert app.interrupt_called is False


class TestModalScreenCtrlDHandling:
    """Tests for app-level Ctrl+D behavior while modals are open."""

    async def test_ctrl_d_deletes_in_thread_selector_instead_of_quitting(self) -> None:
        """App-level quit binding should delegate to thread delete in the modal."""
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        mock_threads: list[ThreadInfo] = [
            {
                "thread_id": "thread-123",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "created_at": "2026-03-08T01:00:00+00:00",
                "initial_prompt": "prompt",
            }
        ]
        with patch(
            "deepagents_cli.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=mock_threads,
        ):
            app = DeepAgentsApp()
            async with app.run_test() as pilot:
                await pilot.pause()

                screen = ThreadSelectorScreen(
                    current_thread=None,
                    initial_threads=mock_threads,
                )
                app.push_screen(screen)
                await pilot.pause()

                with patch.object(app, "exit") as mock_exit:
                    await pilot.press("ctrl+d")
                    await pilot.pause()
                    await pilot.pause()

                assert screen._confirming_delete is True
                mock_exit.assert_not_called()

    async def test_escape_closes_thread_delete_confirm_without_dismissing_modal(
        self,
    ) -> None:
        """Escape should close thread delete confirmation before dismissing modal."""
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        mock_threads: list[ThreadInfo] = [
            {
                "thread_id": "thread-123",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "created_at": "2026-03-08T01:00:00+00:00",
                "initial_prompt": "prompt",
            }
        ]
        with patch(
            "deepagents_cli.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=mock_threads,
        ):
            app = DeepAgentsApp()
            async with app.run_test() as pilot:
                await pilot.pause()

                screen = ThreadSelectorScreen(
                    current_thread=None,
                    initial_threads=mock_threads,
                )
                app.push_screen(screen)
                await pilot.pause()

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert screen.is_delete_confirmation_open is True

                await pilot.press("escape")
                await pilot.pause()
                await pilot.pause()

                assert app.screen is screen
                assert screen.is_delete_confirmation_open is False

    async def test_ctrl_d_twice_quits_from_delete_confirmation(self) -> None:
        """Ctrl+D should use a double-press quit flow inside delete confirmation."""
        from deepagents_cli.widgets.thread_selector import (
            DeleteThreadConfirmScreen,
            ThreadSelectorScreen,
        )

        mock_threads: list[ThreadInfo] = [
            {
                "thread_id": "thread-123",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "created_at": "2026-03-08T01:00:00+00:00",
                "initial_prompt": "prompt",
            }
        ]
        with patch(
            "deepagents_cli.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=mock_threads,
        ):
            app = DeepAgentsApp()
            async with app.run_test() as pilot:
                await pilot.pause()

                screen = ThreadSelectorScreen(
                    current_thread=None,
                    initial_threads=mock_threads,
                )
                app.push_screen(screen)
                await pilot.pause()

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert isinstance(app.screen, DeleteThreadConfirmScreen)

                with (
                    patch.object(app, "notify") as notify_mock,
                    patch.object(app, "exit") as exit_mock,
                ):
                    await pilot.press("ctrl+d")
                    await pilot.pause()
                    notify_mock.assert_called_once_with(
                        "Press Ctrl+D again to quit",
                        timeout=3,
                    )
                    assert app._quit_pending is True
                    exit_mock.assert_not_called()

                    await pilot.press("ctrl+d")
                    await pilot.pause()
                    exit_mock.assert_called_once()

    async def test_ctrl_c_still_works_from_delete_confirmation(self) -> None:
        """Ctrl+C should preserve the normal double-press quit flow in confirmation."""
        from deepagents_cli.widgets.thread_selector import (
            DeleteThreadConfirmScreen,
            ThreadSelectorScreen,
        )

        mock_threads: list[ThreadInfo] = [
            {
                "thread_id": "thread-123",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "created_at": "2026-03-08T01:00:00+00:00",
                "initial_prompt": "prompt",
            }
        ]
        with patch(
            "deepagents_cli.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=mock_threads,
        ):
            app = DeepAgentsApp()
            async with app.run_test() as pilot:
                await pilot.pause()

                screen = ThreadSelectorScreen(
                    current_thread=None,
                    initial_threads=mock_threads,
                )
                app.push_screen(screen)
                await pilot.pause()

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert isinstance(app.screen, DeleteThreadConfirmScreen)

                with (
                    patch.object(app, "notify") as notify_mock,
                    patch.object(app, "exit") as exit_mock,
                ):
                    app.action_quit_or_interrupt()
                    notify_mock.assert_called_once_with(
                        "Press Ctrl+C again to quit",
                        timeout=3,
                    )
                    assert app._quit_pending is True
                    exit_mock.assert_not_called()

                    app.action_quit_or_interrupt()
                    exit_mock.assert_called_once()

    async def test_ctrl_d_quits_from_model_selector_with_input_focused(
        self,
    ) -> None:
        """Ctrl+D should not be swallowed or ignored in the model selector."""
        from deepagents_cli.widgets.model_selector import ModelSelectorScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = ModelSelectorScreen(
                current_model="claude-sonnet-4-5",
                current_provider="anthropic",
            )
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#model-filter", Input)
            assert filter_input.has_focus

            with patch.object(app, "exit") as exit_mock:
                await pilot.press("ctrl+d")
                await pilot.pause()

            exit_mock.assert_called_once()

    async def test_ctrl_d_quits_from_mcp_viewer(self) -> None:
        """Ctrl+D should still quit while the MCP viewer modal is open."""
        from deepagents_cli.mcp_tools import MCPServerInfo, MCPToolInfo
        from deepagents_cli.widgets.mcp_viewer import MCPViewerScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = MCPViewerScreen(
                server_info=[
                    MCPServerInfo(
                        name="filesystem",
                        transport="stdio",
                        tools=[
                            MCPToolInfo(
                                name="read_file",
                                description="Read a file",
                            )
                        ],
                    )
                ]
            )
            app.push_screen(screen)
            await pilot.pause()

            with patch.object(app, "exit") as exit_mock:
                await pilot.press("ctrl+d")
                await pilot.pause()

            exit_mock.assert_called_once()


class TestModalScreenShiftTabHandling:
    """Tests for app-level Shift+Tab behavior while modals are open."""

    async def test_shift_tab_moves_backward_in_thread_selector(self) -> None:
        """Shift+Tab should move backward in the thread selector controls."""
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = ThreadSelectorScreen(
                current_thread=None,
                initial_threads=[
                    {
                        "thread_id": "thread-123",
                        "agent_name": "agent",
                        "updated_at": "2026-03-08T02:00:00+00:00",
                        "created_at": "2026-03-08T01:00:00+00:00",
                        "initial_prompt": "prompt",
                    }
                ],
            )
            app.push_screen(screen)
            await pilot.pause()

            assert app._auto_approve is False
            filter_input = screen.query_one("#thread-filter", Input)
            sort_switch = screen.query_one("#thread-sort-toggle", Checkbox)

            await pilot.press("tab")
            await pilot.pause()
            assert sort_switch.has_focus

            await pilot.press("shift+tab")
            await pilot.pause()

            assert filter_input.has_focus
            assert app._auto_approve is False


class TestModalScreenCtrlCHandling:
    """Tests for app-level Ctrl+C behavior while modals are open."""

    async def test_ctrl_c_quits_from_thread_selector_with_input_focused(
        self,
    ) -> None:
        """Ctrl+C should reach the app even when the thread filter has focus."""
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = ThreadSelectorScreen(
                current_thread=None,
                initial_threads=[
                    {
                        "thread_id": "thread-123",
                        "agent_name": "agent",
                        "updated_at": "2026-03-08T02:00:00+00:00",
                        "created_at": "2026-03-08T01:00:00+00:00",
                        "initial_prompt": "prompt",
                    }
                ],
            )
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#thread-filter", Input)
            assert filter_input.has_focus

            with (
                patch.object(app, "notify") as notify_mock,
                patch.object(app, "exit") as exit_mock,
            ):
                await pilot.press("ctrl+c")
                await pilot.pause()
                notify_mock.assert_called_once_with(
                    "Press Ctrl+C again to quit",
                    timeout=3,
                )
                assert app._quit_pending is True
                exit_mock.assert_not_called()

                await pilot.press("ctrl+c")
                await pilot.pause()
                exit_mock.assert_called_once()

    async def test_ctrl_c_quits_from_model_selector_with_input_focused(
        self,
    ) -> None:
        """Ctrl+C should not be swallowed by the model filter input."""
        from deepagents_cli.widgets.model_selector import ModelSelectorScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = ModelSelectorScreen(
                current_model="claude-sonnet-4-5",
                current_provider="anthropic",
            )
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#model-filter", Input)
            assert filter_input.has_focus

            with (
                patch.object(app, "notify") as notify_mock,
                patch.object(app, "exit") as exit_mock,
            ):
                await pilot.press("ctrl+c")
                await pilot.pause()
                notify_mock.assert_called_once_with(
                    "Press Ctrl+C again to quit",
                    timeout=3,
                )
                assert app._quit_pending is True
                exit_mock.assert_not_called()

                await pilot.press("ctrl+c")
                await pilot.pause()
                exit_mock.assert_called_once()

    async def test_ctrl_c_quits_from_mcp_viewer(self) -> None:
        """Ctrl+C should still trigger app quit flow while the MCP modal is open."""
        from deepagents_cli.mcp_tools import MCPServerInfo, MCPToolInfo
        from deepagents_cli.widgets.mcp_viewer import MCPViewerScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = MCPViewerScreen(
                server_info=[
                    MCPServerInfo(
                        name="filesystem",
                        transport="stdio",
                        tools=[
                            MCPToolInfo(
                                name="read_file",
                                description="Read a file",
                            )
                        ],
                    )
                ]
            )
            app.push_screen(screen)
            await pilot.pause()

            with (
                patch.object(app, "notify") as notify_mock,
                patch.object(app, "exit") as exit_mock,
            ):
                await pilot.press("ctrl+c")
                await pilot.pause()
                notify_mock.assert_called_once_with(
                    "Press Ctrl+C again to quit",
                    timeout=3,
                )
                assert app._quit_pending is True
                exit_mock.assert_not_called()

                await pilot.press("ctrl+c")
                await pilot.pause()
                exit_mock.assert_called_once()


class TestMountMessageNoMatches:
    """Test _mount_message resilience when #messages container is missing.

    When a user interrupts a streaming response, the cancellation handler and
    error handler both call _mount_message. If the screen has been torn down
    (e.g. #messages container no longer exists), this should not crash.
    """

    async def test_mount_message_no_crash_when_messages_missing(self) -> None:
        """_mount_message should not raise NoMatches when #messages is absent."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Verify the #messages container exists initially
            messages_container = app.query_one("#messages", Container)
            assert messages_container is not None

            # Remove #messages to simulate a torn-down screen state
            await messages_container.remove()

            # Verify it's truly gone
            with pytest.raises(NoMatches):
                app.query_one("#messages", Container)

            # _mount_message should handle the missing container gracefully
            # Before the fix, this raises NoMatches
            await app._mount_message(AppMessage("Interrupted by user"))

    async def test_mount_error_message_no_crash_when_messages_missing(
        self,
    ) -> None:
        """ErrorMessage via _mount_message should not crash without #messages.

        This is the second crash in the cascade: after _mount_message fails
        in the CancelledError handler, _run_agent_task's except clause also
        calls _mount_message(ErrorMessage(...)), which fails the same way.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            messages_container = app.query_one("#messages", Container)
            await messages_container.remove()

            # Should not raise
            await app._mount_message(ErrorMessage("Agent error: something"))


class TestQueuedMessage:
    """Test QueuedMessage dataclass."""

    def test_frozen(self) -> None:
        """QueuedMessage should be immutable."""
        msg = QueuedMessage(text="hello", mode="normal")
        with pytest.raises(AttributeError):
            msg.text = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        """QueuedMessage should store text and mode."""
        msg = QueuedMessage(text="hello", mode="shell")
        assert msg.text == "hello"
        assert msg.mode == "shell"


class TestMessageQueue:
    """Test message queue behavior in DeepAgentsApp."""

    async def test_message_queued_when_agent_running(self) -> None:
        """Messages should be queued when agent is running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("queued msg", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "queued msg"
            assert app._pending_messages[0].mode == "normal"

    async def test_message_queued_while_connecting(self) -> None:
        """Messages submitted during server startup should be queued."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            app.post_message(ChatInput.Submitted("early msg", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "early msg"
            widgets = app.query(QueuedUserMessage)
            assert len(widgets) == 1

    async def test_message_blocked_while_thread_switching(self) -> None:
        """Submissions should be ignored while thread switching is in-flight."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._thread_switching = True
            with patch.object(app, "notify") as notify_mock:
                app.post_message(ChatInput.Submitted("blocked msg", "normal"))
                await pilot.pause()

                assert len(app._pending_messages) == 0
                user_msgs = app.query(UserMessage)
                assert not any(w._content == "blocked msg" for w in user_msgs)
                notify_mock.assert_called_once_with(
                    "Thread switch in progress. Please wait.",
                    severity="warning",
                    timeout=3,
                )

    async def test_queued_widget_mounted(self) -> None:
        """Queued messages should produce a QueuedUserMessage widget."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("test msg", "normal"))
            await pilot.pause()

            widgets = app.query(QueuedUserMessage)
            assert len(widgets) == 1
            assert len(app._queued_widgets) == 1

    async def test_immediate_processing_when_agent_idle(self) -> None:
        """Messages should process immediately when agent is not running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert not app._agent_running

            app.post_message(ChatInput.Submitted("direct msg", "normal"))
            await pilot.pause()

            # Should not be queued
            assert len(app._pending_messages) == 0
            # Should be mounted as a regular UserMessage
            user_msgs = app.query(UserMessage)
            assert any(w._content == "direct msg" for w in user_msgs)

    async def test_fifo_order(self) -> None:
        """Queued messages should process in FIFO order."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("first", "normal"))
            await pilot.pause()
            app.post_message(ChatInput.Submitted("second", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 2
            assert app._pending_messages[0].text == "first"
            assert app._pending_messages[1].text == "second"

    async def test_queue_cleared_on_interrupt(self) -> None:
        """Interrupt should clear the message queue."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            # Simulate a worker so action_interrupt has something to cancel
            mock_worker = MagicMock()
            app._agent_worker = mock_worker

            app.post_message(ChatInput.Submitted("msg1", "normal"))
            await pilot.pause()
            app.post_message(ChatInput.Submitted("msg2", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 2

            # Interrupt (escape key handler)
            app.action_interrupt()

            assert len(app._pending_messages) == 0
            assert len(app._queued_widgets) == 0
            mock_worker.cancel.assert_called_once()

    async def test_interrupt_dismisses_completion_without_stopping_agent(self) -> None:
        """Esc should dismiss completion popup without interrupting the agent."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            mock_worker = MagicMock()
            app._agent_worker = mock_worker

            # Activate completion by typing "/"
            chat = app._chat_input
            assert chat is not None
            assert chat._text_area is not None
            chat._text_area.text = "/"
            await pilot.pause()
            assert chat._current_suggestions  # completion is active

            # Esc should dismiss completion, NOT cancel the agent
            app.action_interrupt()

            assert chat._current_suggestions == []
            mock_worker.cancel.assert_not_called()
            assert app._agent_running is True

    async def test_interrupt_falls_through_when_no_completion(self) -> None:
        """Esc should interrupt the agent when completion is not active."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            mock_worker = MagicMock()
            app._agent_worker = mock_worker

            # No completion active — interrupt should reach the agent
            chat = app._chat_input
            assert chat is not None
            assert not chat._current_suggestions

            app.action_interrupt()

            mock_worker.cancel.assert_called_once()

    async def test_queue_cleared_on_ctrl_c(self) -> None:
        """Ctrl+C should clear the message queue."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            mock_worker = MagicMock()
            app._agent_worker = mock_worker

            app.post_message(ChatInput.Submitted("msg", "normal"))
            await pilot.pause()

            app.action_quit_or_interrupt()

            assert len(app._pending_messages) == 0
            assert len(app._queued_widgets) == 0

    async def test_process_next_from_queue_removes_widget(self) -> None:
        """Processing a queued message should remove its ephemeral widget."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Manually enqueue
            app._pending_messages.append(QueuedMessage(text="test", mode="normal"))
            widget = QueuedUserMessage("test")
            messages = app.query_one("#messages", Container)
            await messages.mount(widget)
            app._queued_widgets.append(widget)

            await app._process_next_from_queue()
            await pilot.pause()

            assert len(app._queued_widgets) == 0

    async def test_shell_command_continues_chain(self) -> None:
        """Shell/command messages should not break the queue processing chain."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Queue a shell command followed by a normal message
            app._pending_messages.append(QueuedMessage(text="!echo hi", mode="shell"))
            app._pending_messages.append(
                QueuedMessage(text="hello agent", mode="normal")
            )

            await app._process_next_from_queue()
            await pilot.pause()
            await pilot.pause()

            # The shell command should have been processed and the normal
            # message should also have been picked up (mounted as UserMessage)
            user_msgs = app.query(UserMessage)
            assert any(w._content == "hello agent" for w in user_msgs)


class TestAskUserLifecycle:
    """Tests for ask_user widget cleanup flows."""

    async def test_request_ask_user_timeout_cleans_old_widget(self) -> None:
        """Timeout cleanup should cancel then remove the previous widget."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            old_widget = MagicMock()
            old_widget.remove = AsyncMock()
            app._pending_ask_user_widget = old_widget

            with patch("deepagents_cli.app._monotonic", side_effect=[0.0, 31.0]):
                await app._request_ask_user([{"question": "Name?", "type": "text"}])

            old_widget.action_cancel.assert_called_once()
            old_widget.remove.assert_awaited_once()
            assert old_widget.mock_calls[:2] == [call.action_cancel(), call.remove()]
            assert app._pending_ask_user_widget is not old_widget

    async def test_on_ask_user_menu_answered_ignores_remove_errors(self) -> None:
        """Answered handler should swallow remove races and clear tracking."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            widget = MagicMock()
            widget.remove = AsyncMock(side_effect=RuntimeError("already removed"))
            app._pending_ask_user_widget = widget

            await app.on_ask_user_menu_answered(object())
            await pilot.pause()

            assert app._pending_ask_user_widget is None
            widget.remove.assert_awaited_once()

    async def test_on_ask_user_menu_cancelled_ignores_remove_errors(self) -> None:
        """Cancelled handler should swallow remove races and clear tracking."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            widget = MagicMock()
            widget.remove = AsyncMock(side_effect=RuntimeError("already removed"))
            app._pending_ask_user_widget = widget

            await app.on_ask_user_menu_cancelled(object())
            await pilot.pause()

            assert app._pending_ask_user_widget is None
            widget.remove.assert_awaited_once()


class TestTraceCommand:
    """Test /trace slash command."""

    async def test_trace_opens_browser_when_configured(self) -> None:
        """Should open the LangSmith thread URL in the browser."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = TextualSessionState(thread_id="test-thread-123")

            with (
                patch(
                    "deepagents_cli.app.build_langsmith_thread_url",
                    return_value="https://smith.langchain.com/o/org/projects/p/proj/t/test-thread-123",
                ),
                patch("deepagents_cli.app.webbrowser.open") as mock_open,
            ):
                await app._handle_trace_command("/trace")
                await pilot.pause()

            mock_open.assert_called_once_with(
                "https://smith.langchain.com/o/org/projects/p/proj/t/test-thread-123"
            )
            app_msgs = app.query(AppMessage)
            assert any(  # not a URL check—just verifying the link was rendered
                "https://smith.langchain.com/o/org/projects/p/proj/t/test-thread-123"
                in str(w._content)
                for w in app_msgs
            )

    async def test_trace_shows_error_when_not_configured(self) -> None:
        """Should show configuration hint when LangSmith is not set up."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = TextualSessionState()

            with patch(
                "deepagents_cli.app.build_langsmith_thread_url",
                return_value=None,
            ):
                await app._handle_trace_command("/trace")
                await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("LANGSMITH_API_KEY" in str(w._content) for w in app_msgs)

    async def test_trace_shows_error_when_no_session(self) -> None:
        """Should show error when there is no active session."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = None

            await app._handle_trace_command("/trace")
            await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("No active session" in str(w._content) for w in app_msgs)

    async def test_trace_shows_link_when_browser_fails(self) -> None:
        """Should still display the URL link even if the browser cannot open."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = TextualSessionState(thread_id="test-thread-123")

            with (
                patch(
                    "deepagents_cli.app.build_langsmith_thread_url",
                    return_value="https://smith.langchain.com/t/test-thread-123",
                ),
                patch(
                    "deepagents_cli.app.webbrowser.open",
                    side_effect=webbrowser.Error("no browser"),
                ),
            ):
                await app._handle_trace_command("/trace")
                await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any(  # not a URL check—just verifying the link was rendered
                "https://smith.langchain.com/t/test-thread-123" in str(w._content)
                for w in app_msgs
            )

    async def test_trace_shows_error_when_url_build_raises(self) -> None:
        """Should show error message when build_langsmith_thread_url raises."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = TextualSessionState(thread_id="test-thread-123")

            with patch(
                "deepagents_cli.app.build_langsmith_thread_url",
                side_effect=RuntimeError("SDK error"),
            ):
                await app._handle_trace_command("/trace")
                await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("Failed to resolve" in str(w._content) for w in app_msgs)

    async def test_trace_routed_from_handle_command(self) -> None:
        """'/trace' should be correctly routed through _handle_command."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = None

            await app._handle_command("/trace")
            await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("No active session" in str(w._content) for w in app_msgs)


class TestRunAgentTaskMediaTracker:
    """Tests image tracker wiring from app into textual execution."""

    async def test_run_agent_task_passes_image_tracker(self) -> None:
        """`_run_agent_task` should forward the shared image tracker."""
        app = DeepAgentsApp(agent=MagicMock())
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._ui_adapter is not None

            with patch(
                "deepagents_cli.app.execute_task_textual", new_callable=AsyncMock
            ) as mock_execute:
                await app._run_agent_task("hello")

            mock_execute.assert_awaited_once()
            assert mock_execute.await_args is not None
            assert mock_execute.await_args.kwargs["image_tracker"] is app._image_tracker
            assert mock_execute.await_args.kwargs["sandbox_type"] is app._sandbox_type

    async def test_run_agent_task_finalizes_pending_tools_on_error(self) -> None:
        """Unexpected agent errors should stop/clear in-flight tool widgets."""
        app = DeepAgentsApp(agent=MagicMock())
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._ui_adapter is not None

            pending_tool = MagicMock()
            app._ui_adapter._current_tool_messages = {"tool-1": pending_tool}

            with patch(
                "deepagents_cli.app.execute_task_textual",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ):
                await app._run_agent_task("hello")
                await pilot.pause()

            pending_tool.set_error.assert_called_once_with("Agent error: boom")
            assert app._ui_adapter._current_tool_messages == {}

            errors = app.query(ErrorMessage)
            assert any("Agent error: boom" in str(w._content) for w in errors)


class TestAppFocusRestoresChatInput:
    """Test `on_app_focus` restores chat input focus after terminal regains focus."""

    async def test_app_focus_restores_chat_input(self) -> None:
        """Regaining terminal focus should re-focus the chat input."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._chat_input is not None
            assert app._chat_input._text_area is not None

            # Blur the input to simulate focus loss from webbrowser.open
            app._chat_input._text_area.blur()
            await pilot.pause()

            app.on_app_focus()
            await pilot.pause()

            # chat_input.focus_input should have been called
            assert app._chat_input._text_area.has_focus

    async def test_app_focus_skips_when_modal_open(self) -> None:
        """Regaining focus should not steal focus from an open modal."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Push a modal screen
            from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

            screen = ThreadSelectorScreen(current_thread=None)
            app.push_screen(screen)
            await pilot.pause()

            assert isinstance(app.screen, ModalScreen)

            # on_app_focus should be a no-op with modal open
            with patch.object(app._chat_input, "focus_input") as mock_focus:
                app.on_app_focus()

            mock_focus.assert_not_called()

    async def test_app_focus_skips_when_approval_pending(self) -> None:
        """Regaining focus should not steal focus from the approval widget."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._chat_input is not None

            # Simulate a pending approval widget
            app._pending_approval_widget = MagicMock()

            with patch.object(app._chat_input, "focus_input") as mock_focus:
                app.on_app_focus()

            mock_focus.assert_not_called()


class TestPasteRouting:
    """Tests app-level paste routing when chat input focus lags."""

    async def test_on_paste_routes_unfocused_event_to_chat_input(self) -> None:
        """Unfocused paste events should be forwarded to chat input handler."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._chat_input is not None

            event = events.Paste("/tmp/photo.png")
            with (
                patch.object(app, "_is_input_focused", return_value=False),
                patch.object(
                    app._chat_input, "handle_external_paste", return_value=True
                ) as mock_handle,
                patch.object(event, "prevent_default") as mock_prevent,
                patch.object(event, "stop") as mock_stop,
            ):
                app.on_paste(event)

            mock_handle.assert_called_once_with("/tmp/photo.png")
            mock_prevent.assert_called_once()
            mock_stop.assert_called_once()

    async def test_on_paste_does_not_route_when_input_already_focused(self) -> None:
        """Focused input should keep normal TextArea paste handling path."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._chat_input is not None

            event = events.Paste("/tmp/photo.png")
            with (
                patch.object(app, "_is_input_focused", return_value=True),
                patch.object(
                    app._chat_input, "handle_external_paste", return_value=True
                ) as mock_handle,
                patch.object(event, "prevent_default") as mock_prevent,
                patch.object(event, "stop") as mock_stop,
            ):
                app.on_paste(event)

            mock_handle.assert_not_called()
            mock_prevent.assert_not_called()
            mock_stop.assert_not_called()


class TestShellCommandInterrupt:
    """Tests for interruptible shell commands (! prefix) using worker pattern."""

    async def test_escape_cancels_shell_worker(self) -> None:
        """Esc while shell command is running should cancel the worker."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            mock_worker = MagicMock()
            app._shell_worker = mock_worker

            app.action_interrupt()

            mock_worker.cancel.assert_called_once()
            assert len(app._pending_messages) == 0

    async def test_ctrl_c_cancels_shell_worker(self) -> None:
        """Ctrl+C while shell command is running should cancel the worker."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            mock_worker = MagicMock()
            app._shell_worker = mock_worker

            # Queue a message to verify it gets cleared
            app._pending_messages.append(QueuedMessage(text="queued", mode="normal"))

            app.action_quit_or_interrupt()

            mock_worker.cancel.assert_called_once()
            assert len(app._pending_messages) == 0
            assert app._quit_pending is False

    async def test_process_killed_on_cancelled_error(self) -> None:
        """CancelledError in _run_shell_task should kill the process."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.CancelledError)
            mock_proc.returncode = None
            mock_proc.pid = 12345
            mock_proc.wait = AsyncMock()

            with (
                patch(
                    "asyncio.create_subprocess_shell",
                    return_value=mock_proc,
                ),
                patch("os.killpg") as mock_killpg,
                patch("os.getpgid", return_value=12345),
                pytest.raises(asyncio.CancelledError),
            ):
                await app._run_shell_task("sleep 999")

            mock_killpg.assert_called()

    async def test_cleanup_clears_state(self) -> None:
        """_cleanup_shell_task should reset all shell state."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            app._shell_worker = MagicMock()
            app._shell_worker.is_cancelled = False
            app._shell_process = None

            await app._cleanup_shell_task()

            assert app._shell_process is None
            assert app._shell_running is False
            assert app._shell_worker is None

    async def test_messages_queued_during_shell(self) -> None:
        """Messages should be queued while shell command runs."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._shell_running = True

            app.post_message(ChatInput.Submitted("queued msg", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "queued msg"

    async def test_queue_drains_after_shell_completes(self) -> None:
        """Pending messages should drain after _cleanup_shell_task."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            app._shell_worker = MagicMock()
            app._shell_worker.is_cancelled = False
            app._shell_process = None

            # Enqueue a message
            app._pending_messages.append(
                QueuedMessage(text="after shell", mode="normal")
            )

            await app._cleanup_shell_task()
            await pilot.pause()

            # Message should have been processed (mounted as UserMessage)
            user_msgs = app.query(UserMessage)
            assert any(w._content == "after shell" for w in user_msgs)

    async def test_interrupted_shows_message(self) -> None:
        """Cancelled worker should show 'Command interrupted'."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            mock_worker = MagicMock()
            mock_worker.is_cancelled = True
            app._shell_worker = mock_worker
            # Process still set means it was interrupted mid-flight
            mock_proc = MagicMock()
            mock_proc.returncode = None
            app._shell_process = mock_proc

            await app._cleanup_shell_task()
            await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("Command interrupted" in str(w._content) for w in app_msgs)

    async def test_timeout_kills_and_shows_error(self) -> None:
        """Timeout in _run_shell_task should kill process and show error."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.returncode = None
            mock_proc.pid = 12345
            mock_proc.wait = AsyncMock()

            with (
                patch(
                    "asyncio.create_subprocess_shell",
                    return_value=mock_proc,
                ),
                patch("os.killpg"),
                patch("os.getpgid", return_value=12345),
            ):
                await app._run_shell_task("sleep 999")
                await pilot.pause()

            assert app._shell_process is None
            error_msgs = app.query(ErrorMessage)
            assert any("timed out" in w._content for w in error_msgs)

    async def test_posix_killpg_called(self) -> None:
        """On POSIX, _kill_shell_process should use os.killpg with SIGTERM."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.pid = 42
            mock_proc.wait = AsyncMock()
            app._shell_process = mock_proc

            with (
                patch("deepagents_cli.app.sys") as mock_sys,
                patch("os.killpg") as mock_killpg,
                patch("os.getpgid", return_value=42) as mock_getpgid,
            ):
                mock_sys.platform = "linux"
                await app._kill_shell_process()

            mock_getpgid.assert_called_once_with(42)
            mock_killpg.assert_called_once_with(42, signal.SIGTERM)

    async def test_sigkill_escalation(self) -> None:
        """SIGKILL should be sent when SIGTERM times out."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.pid = 42
            mock_proc.wait = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.kill = MagicMock()
            app._shell_process = mock_proc

            with (
                patch("deepagents_cli.app.sys") as mock_sys,
                patch("os.killpg") as mock_killpg,
                patch("os.getpgid", return_value=42),
            ):
                mock_sys.platform = "linux"
                await app._kill_shell_process()

            # First call: SIGTERM, second call: SIGKILL
            assert mock_killpg.call_count == 2
            mock_killpg.assert_any_call(42, signal.SIGTERM)
            mock_killpg.assert_any_call(42, signal.SIGKILL)

    async def test_no_op_when_no_shell_running(self) -> None:
        """Ctrl+C with no shell command running should fall through to quit hint."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            assert not app._shell_running
            app.action_quit_or_interrupt()

            assert app._quit_pending is True

    async def test_oserror_shows_error_message(self) -> None:
        """OSError from create_subprocess_shell should display error."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            with patch(
                "asyncio.create_subprocess_shell",
                side_effect=OSError("Permission denied"),
            ):
                await app._run_shell_task("forbidden")
                await pilot.pause()

            assert app._shell_process is None
            error_msgs = app.query(ErrorMessage)
            assert any("Permission denied" in w._content for w in error_msgs)

    async def test_handle_shell_command_sets_running_state(self) -> None:
        """_handle_shell_command should set _shell_running and spawn worker."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            with patch.object(app, "run_worker") as mock_rw:
                mock_rw.return_value = MagicMock()
                await app._handle_shell_command("echo hi")

            assert app._shell_running is True
            assert app._shell_worker is not None
            mock_rw.assert_called_once()
            # Close the unawaited coroutine to suppress RuntimeWarning
            coro = mock_rw.call_args[0][0]
            coro.close()

    async def test_kill_noop_when_already_exited(self) -> None:
        """_kill_shell_process should no-op if process already exited."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.pid = 42
            app._shell_process = mock_proc

            with patch("os.killpg") as mock_killpg:
                await app._kill_shell_process()

            mock_killpg.assert_not_called()
            mock_proc.terminate.assert_not_called()

    async def test_end_to_end_escape_during_shell(self) -> None:
        """Esc during a running shell worker should cancel execution."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Simulate a running shell state with a mock worker
            app._shell_running = True
            mock_worker = MagicMock()
            app._shell_worker = mock_worker

            await pilot.press("escape")
            await pilot.pause()

            mock_worker.cancel.assert_called_once()


class TestInterruptApprovalPriority:
    """Tests for escape interrupt priority when HITL approval is pending."""

    async def test_escape_rejects_approval_before_canceling_worker(self) -> None:
        """When both HITL approval and worker are active, reject approval first."""
        app = DeepAgentsApp()
        approval = MagicMock()
        worker = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            app._pending_approval_widget = approval
            app._agent_running = True
            app._agent_worker = worker

            app.action_interrupt()

        approval.action_select_reject.assert_called_once()
        worker.cancel.assert_not_called()

    async def test_escape_cancels_worker_when_no_approval_pending(self) -> None:
        """Escape cancels active worker and clears queued messages when no approval."""
        app = DeepAgentsApp()
        worker = MagicMock()
        queued_w1 = MagicMock()
        queued_w2 = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            app._pending_approval_widget = None
            app._agent_running = True
            app._agent_worker = worker
            app._pending_messages.append(QueuedMessage(text="q", mode="normal"))
            app._queued_widgets.append(queued_w1)
            app._queued_widgets.append(queued_w2)

            app.action_interrupt()

        worker.cancel.assert_called_once()
        queued_w1.remove.assert_called_once()
        queued_w2.remove.assert_called_once()
        assert len(app._pending_messages) == 0
        assert len(app._queued_widgets) == 0

    async def test_escape_rejects_approval_when_no_worker(self) -> None:
        """Approval rejection works even without an active agent worker."""
        app = DeepAgentsApp()
        approval = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            app._pending_approval_widget = approval
            app._agent_running = False
            app._agent_worker = None

            app.action_interrupt()

        approval.action_select_reject.assert_called_once()

    async def test_ctrl_c_rejects_approval_before_canceling_worker(self) -> None:
        """Ctrl+C should also reject approval before canceling worker."""
        app = DeepAgentsApp()
        approval = MagicMock()
        worker = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            app._pending_approval_widget = approval
            app._agent_running = True
            app._agent_worker = worker

            app.action_quit_or_interrupt()

        approval.action_select_reject.assert_called_once()
        worker.cancel.assert_not_called()
        assert app._quit_pending is False


class TestIsUserTyping:
    """Unit tests for `_is_user_typing()` threshold logic."""

    def test_returns_false_when_never_typed(self) -> None:
        """Should return False if _last_typed_at is None."""
        app = DeepAgentsApp()
        assert app._is_user_typing() is False

    def test_returns_true_within_threshold(self) -> None:
        """Should return True right after a keystroke."""
        app = DeepAgentsApp()
        app._last_typed_at = time.monotonic()
        assert app._is_user_typing() is True

    def test_returns_false_after_threshold(self) -> None:
        """Should return False once the idle threshold has elapsed."""
        app = DeepAgentsApp()
        app._last_typed_at = time.monotonic() - (_TYPING_IDLE_THRESHOLD_SECONDS + 0.1)
        assert app._is_user_typing() is False

    def test_boundary_just_within_threshold(self) -> None:
        """Should return True when just inside the threshold window."""
        app = DeepAgentsApp()
        app._last_typed_at = time.monotonic() - (_TYPING_IDLE_THRESHOLD_SECONDS - 0.1)
        assert app._is_user_typing() is True


class TestRequestApprovalBranching:
    """_request_approval should show a placeholder when the user is typing."""

    async def test_placeholder_mounted_when_typing(self) -> None:
        """If the user is typing, a Static placeholder is mounted instead of menu."""
        app = DeepAgentsApp(agent=MagicMock())
        # Simulate recent typing
        app._last_typed_at = time.monotonic()

        mounted_classes: list[str] = []

        async def fake_mount_before_queued(  # noqa: RUF029
            _container: object, widget: object
        ) -> None:
            if isinstance(widget, Static):
                mounted_classes.append(" ".join(widget.classes))

        app._mount_before_queued = fake_mount_before_queued  # type: ignore[assignment]

        # Prevent actual worker from running; we just want to check branching.
        run_worker_calls: list[object] = []

        def _stub_worker(coro: object, **_: object) -> MagicMock:
            # Consume the coroutine immediately to suppress RuntimeWarning.
            if inspect.iscoroutine(coro):
                coro.close()
            run_worker_calls.append(coro)
            return MagicMock()

        app.run_worker = _stub_worker  # type: ignore[method-assign]

        dummy_container = MagicMock()
        app.query_one = MagicMock(return_value=dummy_container)  # type: ignore[method-assign]

        action_requests = [
            {"name": "write_file", "args": {"path": "/tmp/x.txt", "content": "hi"}}
        ]
        future = asyncio.get_running_loop().create_future()

        with patch.object(asyncio, "get_running_loop") as mock_loop:
            mock_loop.return_value.create_future.return_value = future
            returned = await app._request_approval(action_requests, None)

        assert returned is future
        assert any("approval-placeholder" in cls for cls in mounted_classes), (
            f"Expected 'approval-placeholder' in mounted widget classes,"
            f" got {mounted_classes}"
        )
        assert len(run_worker_calls) == 1, (
            "run_worker should have been called once for the deferred swap"
        )

    async def test_placeholder_mount_failure_falls_back_to_menu(self) -> None:
        """If placeholder mount fails, the ApprovalMenu is shown directly."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        app._last_typed_at = time.monotonic()

        mounted_types: list[type] = []

        call_count = 0

        async def failing_then_ok_mount(  # noqa: RUF029
            _container: object, widget: object
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "simulated mount failure"
                raise RuntimeError(msg)
            mounted_types.append(type(widget))

        app._mount_before_queued = failing_then_ok_mount  # type: ignore[assignment]
        app.call_after_refresh = MagicMock()  # type: ignore[method-assign]

        dummy_container = MagicMock()
        app.query_one = MagicMock(return_value=dummy_container)  # type: ignore[method-assign]

        action_requests = [
            {"name": "write_file", "args": {"path": "/tmp/z.txt", "content": "hi"}}
        ]
        future = asyncio.get_running_loop().create_future()

        with patch.object(asyncio, "get_running_loop") as mock_loop:
            mock_loop.return_value.create_future.return_value = future
            returned = await app._request_approval(action_requests, None)

        assert returned is future
        # Placeholder mount (1st call) fails, fallback menu mount (2nd call)
        # succeeds. The menu is now mounted and the future awaits user input.
        assert ApprovalMenu in mounted_types, (
            f"Expected ApprovalMenu fallback mount, got {mounted_types}"
        )

    async def test_menu_mounted_directly_when_not_typing(self) -> None:
        """If the user is NOT typing, the ApprovalMenu is mounted directly."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        app._last_typed_at = None

        mounted_types: list[type] = []

        async def fake_mount_before_queued(  # noqa: RUF029
            _container: object, widget: object
        ) -> None:
            mounted_types.append(type(widget))

        app._mount_before_queued = fake_mount_before_queued  # type: ignore[assignment]
        app.call_after_refresh = MagicMock()  # type: ignore[method-assign]

        dummy_container = MagicMock()
        app.query_one = MagicMock(return_value=dummy_container)  # type: ignore[method-assign]

        action_requests = [
            {"name": "write_file", "args": {"path": "/tmp/y.txt", "content": "hi"}}
        ]
        future = asyncio.get_running_loop().create_future()

        with patch.object(asyncio, "get_running_loop") as mock_loop:
            mock_loop.return_value.create_future.return_value = future
            returned = await app._request_approval(action_requests, None)

        assert returned is future
        assert ApprovalMenu in mounted_types, (
            f"Expected ApprovalMenu to be mounted, got {mounted_types}"
        )


class TestDeferredShowApproval:
    """_deferred_show_approval should swap placeholder once idle."""

    async def test_swaps_placeholder_for_menu_after_idle(self) -> None:
        """Once typing stops, placeholder is removed and menu is mounted."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        app._last_typed_at = time.monotonic()

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = True
        remove_called = False

        async def fake_remove() -> None:  # noqa: RUF029
            nonlocal remove_called
            remove_called = True

        placeholder.remove = fake_remove

        action_requests = [{"name": "write_file", "args": {}}]
        future = asyncio.get_running_loop().create_future()
        menu = ApprovalMenu(action_requests[0])
        menu.set_future(future)

        mount_called = False

        async def fake_mount_approval(  # noqa: RUF029
            m: ApprovalMenu,  # noqa: ARG001
            f: asyncio.Future[dict[str, str]],  # noqa: ARG001
        ) -> None:
            nonlocal mount_called
            mount_called = True

        app._mount_approval_widget = fake_mount_approval  # type: ignore[method-assign]

        async def stop_typing() -> None:
            await asyncio.sleep(0.05)
            app._last_typed_at = None

        typing_task = asyncio.create_task(stop_typing())
        await app._deferred_show_approval(placeholder, menu, future)
        await typing_task

        assert remove_called, "placeholder.remove() should have been called"
        assert mount_called, "_mount_approval_widget should have been called"

    async def test_bails_if_placeholder_detached_and_cancels_future(self) -> None:
        """If placeholder is detached, worker cancels the future and exits."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        app._last_typed_at = None

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = False

        mount_called = False

        async def fake_mount_approval(  # noqa: RUF029
            m: ApprovalMenu,  # noqa: ARG001
            f: asyncio.Future[dict[str, str]],  # noqa: ARG001
        ) -> None:
            nonlocal mount_called
            mount_called = True

        app._mount_approval_widget = fake_mount_approval  # type: ignore[method-assign]

        action_requests = [{"name": "shell", "args": {"command": "ls"}}]
        future = asyncio.get_running_loop().create_future()
        menu = ApprovalMenu(action_requests[0])
        menu.set_future(future)

        await app._deferred_show_approval(placeholder, menu, future)

        assert not mount_called, "_mount_approval_widget should NOT have been called"
        assert future.cancelled(), "future should have been cancelled"
        assert app._pending_approval_widget is None
        assert app._approval_placeholder is None

    async def test_timeout_shows_approval_after_deadline(self) -> None:
        """If the user types continuously past the deadline, menu is shown anyway."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        # Simulate user typing *forever* by keeping _last_typed_at fresh
        app._last_typed_at = time.monotonic()

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = True

        remove_called = False

        async def fake_remove() -> None:  # noqa: RUF029
            nonlocal remove_called
            remove_called = True

        placeholder.remove = fake_remove

        mount_called = False

        async def fake_mount_approval(  # noqa: RUF029
            m: ApprovalMenu,  # noqa: ARG001
            f: asyncio.Future[dict[str, str]],  # noqa: ARG001
        ) -> None:
            nonlocal mount_called
            mount_called = True

        app._mount_approval_widget = fake_mount_approval  # type: ignore[method-assign]

        action_requests = [{"name": "write_file", "args": {}}]
        future = asyncio.get_running_loop().create_future()
        menu = ApprovalMenu(action_requests[0])
        menu.set_future(future)

        # Patch the timeout to be tiny so the test doesn't actually wait 30s
        with patch("deepagents_cli.app._DEFERRED_APPROVAL_TIMEOUT_SECONDS", 0.05):
            await app._deferred_show_approval(placeholder, menu, future)

        assert remove_called, "placeholder.remove() should have been called"
        assert mount_called, (
            "_mount_approval_widget should have been called after timeout"
        )


class TestOnChatInputTyping:
    """on_chat_input_typing should set _last_typed_at."""

    def test_sets_last_typed_at(self) -> None:
        """Calling on_chat_input_typing records a recent monotonic time."""
        app = DeepAgentsApp()
        assert app._last_typed_at is None

        event = MagicMock()
        before = time.monotonic()
        app.on_chat_input_typing(event)
        after = time.monotonic()

        assert app._last_typed_at is not None
        assert before <= app._last_typed_at <= after

    def test_updates_on_subsequent_calls(self) -> None:
        """Each call should update _last_typed_at to a newer timestamp."""
        app = DeepAgentsApp()
        event = MagicMock()

        app.on_chat_input_typing(event)
        first = app._last_typed_at

        app.on_chat_input_typing(event)
        second = app._last_typed_at

        assert second is not None
        assert first is not None
        assert second >= first


class TestOnApprovalMenuDecidedCleanup:
    """on_approval_menu_decided should defensively clean up placeholders."""

    async def test_removes_attached_placeholder(self) -> None:
        """An attached placeholder should be removed and nulled."""
        app = DeepAgentsApp(agent=MagicMock())

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = True
        remove_called = False

        async def fake_remove() -> None:  # noqa: RUF029
            nonlocal remove_called
            remove_called = True

        placeholder.remove = fake_remove
        app._approval_placeholder = placeholder
        app._pending_approval_widget = None

        event = MagicMock()
        app._chat_input = None
        await app.on_approval_menu_decided(event)

        assert remove_called
        assert app._approval_placeholder is None

    async def test_nulls_detached_placeholder(self) -> None:
        """A detached placeholder should be nulled without calling remove."""
        app = DeepAgentsApp(agent=MagicMock())

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = False
        app._approval_placeholder = placeholder
        app._pending_approval_widget = None

        event = MagicMock()
        app._chat_input = None
        await app.on_approval_menu_decided(event)

        assert app._approval_placeholder is None
        placeholder.remove.assert_not_called()

    async def test_no_placeholder_works_normally(self) -> None:
        """When no placeholder exists, handler proceeds without error."""
        app = DeepAgentsApp(agent=MagicMock())
        app._approval_placeholder = None
        app._pending_approval_widget = None

        event = MagicMock()
        app._chat_input = None
        await app.on_approval_menu_decided(event)

        assert app._approval_placeholder is None


class TestActionOpenEditor:
    """Tests for the external editor action."""

    async def test_updates_text_on_successful_edit(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        text_area = MagicMock()
        text_area.text = "original"
        chat_input = MagicMock()
        chat_input._text_area = text_area
        app._chat_input = chat_input

        with (
            patch.object(app, "suspend"),
            patch("deepagents_cli.editor.open_in_editor", return_value="edited"),
        ):
            await app.action_open_editor()

        assert text_area.text == "edited"
        chat_input.focus_input.assert_called_once()

    async def test_no_update_when_editor_returns_none(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        text_area = MagicMock()
        text_area.text = "original"
        chat_input = MagicMock()
        chat_input._text_area = text_area
        app._chat_input = chat_input

        with (
            patch.object(app, "suspend"),
            patch("deepagents_cli.editor.open_in_editor", return_value=None),
        ):
            await app.action_open_editor()

        assert text_area.text == "original"
        chat_input.focus_input.assert_called_once()

    async def test_early_return_when_chat_input_is_none(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        app._chat_input = None

        # Should not raise
        await app.action_open_editor()

    async def test_early_return_when_text_area_is_none(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        chat_input = MagicMock()
        chat_input._text_area = None
        app._chat_input = chat_input

        await app.action_open_editor()

    async def test_notifies_on_exception(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        text_area = MagicMock()
        text_area.text = ""
        chat_input = MagicMock()
        chat_input._text_area = text_area
        app._chat_input = chat_input

        with (
            patch.object(app, "suspend"),
            patch(
                "deepagents_cli.editor.open_in_editor",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(app, "notify") as mock_notify,
        ):
            await app.action_open_editor()

        mock_notify.assert_called_once()
        assert "failed" in mock_notify.call_args[0][0].lower()
        chat_input.focus_input.assert_called_once()


class TestEditorSlashCommand:
    """Test that /editor dispatches to action_open_editor."""

    async def test_editor_command_calls_action(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        with patch.object(app, "action_open_editor", new_callable=AsyncMock) as mock:
            app._chat_input = MagicMock()
            await app._handle_command("/editor")
        mock.assert_awaited_once()


class TestFetchThreadHistoryData:
    """Verify _fetch_thread_history_data handles server-mode resume scenarios."""

    async def test_dict_messages_converted_to_message_objects(self) -> None:
        """Dict-based messages from server mode are deserialized before conversion."""
        from deepagents_cli.widgets.message_store import MessageData, MessageType

        state = MagicMock()
        state.values = {
            "messages": [
                {"type": "human", "content": "hello", "id": "h1"},
                {
                    "type": "ai",
                    "content": "Hi there!",
                    "id": "a1",
                    "tool_calls": [],
                },
            ],
        }

        mock_agent = AsyncMock()
        mock_agent.aget_state.return_value = state

        app = DeepAgentsApp(agent=mock_agent, thread_id="t-1")
        result = await app._fetch_thread_history_data("t-1")

        assert len(result) == 2
        assert isinstance(result[0], MessageData)
        assert result[0].type == MessageType.USER
        assert result[0].content == "hello"
        assert isinstance(result[1], MessageData)
        assert result[1].type == MessageType.ASSISTANT
        assert result[1].content == "Hi there!"

    async def test_server_mode_falls_back_to_checkpointer(self) -> None:
        """When the server returns empty state, read SQLite checkpointer directly."""
        from langchain_core.messages import AIMessage, HumanMessage

        from deepagents_cli.remote_client import RemoteAgent
        from deepagents_cli.widgets.message_store import MessageData, MessageType

        # Server returns empty state (fresh restart, thread not loaded)
        empty_state = MagicMock()
        empty_state.values = {}

        # spec=RemoteAgent so _remote_agent() isinstance check passes
        mock_agent = MagicMock(spec=RemoteAgent)
        mock_agent.aget_state = AsyncMock(return_value=empty_state)

        app = DeepAgentsApp(agent=mock_agent, thread_id="t-1")

        # Patch the checkpointer fallback to return messages
        checkpointer_msgs = [
            HumanMessage(content="hello", id="h1"),
            AIMessage(content="world", id="a1"),
        ]
        with patch.object(
            DeepAgentsApp,
            "_read_channel_values_from_checkpointer",
            return_value={"messages": checkpointer_msgs},
        ):
            result = await app._fetch_thread_history_data("t-1")

        assert len(result) == 2
        assert result[0].type == MessageType.USER
        assert result[0].content == "hello"
        assert result[1].type == MessageType.ASSISTANT
        assert result[1].content == "world"


class TestRemoteAgent:
    """Tests for DeepAgentsApp._remote_agent()."""

    def test_returns_instance_with_remote_agent(self) -> None:
        from deepagents_cli.remote_client import RemoteAgent

        app = DeepAgentsApp()
        agent = RemoteAgent("http://test:0")
        app._agent = agent
        assert app._remote_agent() is agent

    def test_none_when_agent_is_none(self) -> None:
        app = DeepAgentsApp()
        assert app._remote_agent() is None

    def test_none_with_non_remote_agent(self) -> None:
        """Local Pregel-like agent returns None."""
        app = DeepAgentsApp()
        app._agent = MagicMock()
        assert app._remote_agent() is None

    def test_none_with_mock_spec_pregel(self) -> None:
        """MagicMock without RemoteAgent spec returns None."""
        app = DeepAgentsApp()
        app._agent = MagicMock(spec=[])
        assert app._remote_agent() is None


class TestSlashCommandBypass:
    """Test that certain slash commands bypass the queue gate."""

    async def test_quit_bypasses_queue_when_agent_running(self) -> None:
        """/quit should exit immediately even when agent is running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            with patch.object(app, "exit") as exit_mock:
                app.post_message(ChatInput.Submitted("/quit", "command"))
                await pilot.pause()

            exit_mock.assert_called_once()
            assert len(app._pending_messages) == 0

    async def test_quit_bypasses_queue_when_connecting(self) -> None:
        """/quit should exit immediately even when connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            with patch.object(app, "exit") as exit_mock:
                app.post_message(ChatInput.Submitted("/quit", "command"))
                await pilot.pause()

            exit_mock.assert_called_once()
            assert len(app._pending_messages) == 0

    async def test_quit_bypasses_thread_switching(self) -> None:
        """/quit should exit even during a thread switch."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._thread_switching = True

            with patch.object(app, "exit") as exit_mock:
                app.post_message(ChatInput.Submitted("/quit", "command"))
                await pilot.pause()

            exit_mock.assert_called_once()

    async def test_q_alias_bypasses_queue(self) -> None:
        """/q alias should also bypass the queue."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            with patch.object(app, "exit") as exit_mock:
                app.post_message(ChatInput.Submitted("/q", "command"))
                await pilot.pause()

            exit_mock.assert_called_once()
            assert len(app._pending_messages) == 0

    async def test_version_executes_during_connecting(self) -> None:
        """/version should process immediately when only connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/version", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/version", "command")
            assert len(app._pending_messages) == 0

    async def test_version_queues_during_agent_running(self) -> None:
        """/version should still queue when agent is actively running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("/version", "command"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "/version"

    async def test_model_no_args_opens_selector_during_agent_running(self) -> None:
        """/model (no args) should process immediately during agent run."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/model", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/model", "command")
            assert len(app._pending_messages) == 0

    async def test_model_no_args_opens_selector_during_connecting(self) -> None:
        """/model (no args) should process immediately during connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/model", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/model", "command")

    async def test_model_with_args_still_queues(self) -> None:
        """/model <name> (with args) should still queue normally."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("/model gpt-4", "command"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "/model gpt-4"

    async def test_threads_opens_selector_during_agent_running(self) -> None:
        """/threads should process immediately during agent run."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/threads", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/threads", "command")
            assert len(app._pending_messages) == 0

    async def test_threads_opens_selector_during_connecting(self) -> None:
        """/threads should process immediately during connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/threads", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/threads", "command")


class TestDeferredActions:
    """Test deferred action queueing and draining."""

    async def test_deferred_actions_drain_after_agent_cleanup(self) -> None:
        """Deferred actions should execute when agent task completes."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed = []

            async def action() -> None:
                executed.append("ran")

            app._deferred_actions.append(action)
            app._agent_running = True

            # Simulate agent finishing
            await app._cleanup_agent_task()

            assert executed == ["ran"]
            assert len(app._deferred_actions) == 0

    async def test_deferred_actions_drain_after_shell_cleanup(self) -> None:
        """Deferred actions should execute when shell task completes."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed = []

            async def action() -> None:
                executed.append("ran")

            app._deferred_actions.append(action)
            app._shell_running = True

            await app._cleanup_shell_task()

            assert executed == ["ran"]
            assert len(app._deferred_actions) == 0

    async def test_deferred_actions_not_drained_while_connecting(self) -> None:
        """Deferred actions should NOT drain if still connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed = []

            async def action() -> None:
                executed.append("ran")

            app._deferred_actions.append(action)
            app._agent_running = True
            app._connecting = True

            await app._cleanup_agent_task()

            assert executed == []
            assert len(app._deferred_actions) == 1

    async def test_deferred_actions_cleared_on_interrupt(self) -> None:
        """Deferred actions should be cleared when queue is discarded."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def action() -> None:
                pass

            app._deferred_actions.append(action)
            app._discard_queue()

            assert len(app._deferred_actions) == 0

    async def test_deferred_actions_cleared_on_server_failure(self) -> None:
        """Deferred actions should be cleared when server startup fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def action() -> None:
                pass

            app._deferred_actions.append(action)
            app._connecting = True

            app.on_deep_agents_app_server_start_failed(
                DeepAgentsApp.ServerStartFailed(error=RuntimeError("test"))
            )

            assert len(app._deferred_actions) == 0

    async def test_failing_deferred_action_does_not_block_others(self) -> None:
        """A failing deferred action should not prevent subsequent ones."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed = []

            async def bad_action() -> None:
                raise RuntimeError("boom")

            async def good_action() -> None:
                executed.append("ok")

            app._deferred_actions.append(bad_action)
            app._deferred_actions.append(good_action)

            await app._drain_deferred_actions()

            assert executed == ["ok"]
            assert len(app._deferred_actions) == 0
