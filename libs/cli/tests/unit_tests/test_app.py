"""Unit tests for DeepAgentsApp."""

import io
import os
import webbrowser
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_cli.app import (
    _ITERM_CURSOR_GUIDE_OFF,
    _ITERM_CURSOR_GUIDE_ON,
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


class TestAppBindings:
    """Test app keybindings."""

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
    @pytest.mark.asyncio
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


class TestMountMessageNoMatches:
    """Test _mount_message resilience when #messages container is missing.

    When a user interrupts a streaming response, the cancellation handler and
    error handler both call _mount_message. If the screen has been torn down
    (e.g. #messages container no longer exists), this should not crash.
    """

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
        msg = QueuedMessage(text="hello", mode="bash")
        assert msg.text == "hello"
        assert msg.mode == "bash"


class TestMessageQueue:
    """Test message queue behavior in DeepAgentsApp."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_bash_command_continues_chain(self) -> None:
        """Bash/command messages should not break the queue processing chain."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Queue a bash command followed by a normal message
            app._pending_messages.append(QueuedMessage(text="!echo hi", mode="bash"))
            app._pending_messages.append(
                QueuedMessage(text="hello agent", mode="normal")
            )

            await app._process_next_from_queue()
            await pilot.pause()
            await pilot.pause()

            # The bash command should have been processed and the normal
            # message should also have been picked up (mounted as UserMessage)
            user_msgs = app.query(UserMessage)
            assert any(w._content == "hello agent" for w in user_msgs)


class TestTraceCommand:
    """Test /trace slash command."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


class TestRunAgentTaskImageTracker:
    """Tests image tracker wiring from app into textual execution."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
