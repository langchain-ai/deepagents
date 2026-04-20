"""Tests for non-interactive mode HITL decision logic."""

import io
import sys
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
from rich.style import Style
from rich.text import Text

from deepagents_cli.config import SHELL_ALLOW_ALL, ModelResult
from deepagents_cli.non_interactive import (
    _MAX_HITL_ITERATIONS,
    HITLIterationLimitError,
    ThreadUrlLookupState,
    _build_non_interactive_header,
    _collect_action_request_warnings,
    _make_hitl_decision,
    _run_agent_loop,
    _start_langsmith_thread_url_lookup,
    run_non_interactive,
)


@pytest.fixture
def console() -> Console:
    """Console that captures output."""
    return Console(quiet=True)


class TestMakeHitlDecision:
    """Tests for _make_hitl_decision()."""

    def test_non_shell_action_approved(self, console: Console) -> None:
        """Non-shell actions should be auto-approved."""
        result = _make_hitl_decision(
            {"name": "read_file", "args": {"path": "/tmp/test"}}, console
        )
        assert result == {"type": "approve"}

    def test_shell_without_allow_list_rejected(self, console: Console) -> None:
        """Shell commands should be rejected when no allow-list is configured."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = None
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "rm -rf /"}}, console
            )
            assert result["type"] == "reject"
            assert "not permitted" in result["message"]

    def test_shell_allowed_command_approved(self, console: Console) -> None:
        """Shell commands in the allow-list should be approved."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls", "cat", "grep"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "ls -la"}}, console
            )
            assert result == {"type": "approve"}

    def test_shell_disallowed_command_rejected(self, console: Console) -> None:
        """Shell commands not in the allow-list should be rejected."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls", "cat", "grep"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "rm -rf /"}}, console
            )
            assert result["type"] == "reject"
            assert "rm -rf /" in result["message"]
            assert "not in the allow-list" in result["message"]

    def test_shell_rejected_message_includes_allowed_commands(
        self, console: Console
    ) -> None:
        """Rejection message should list the allowed commands."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls", "cat"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "whoami"}}, console
            )
            assert "ls" in result["message"]
            assert "cat" in result["message"]

    def test_empty_action_name_approved(self, console: Console) -> None:
        """Actions with empty name should be approved (non-shell)."""
        result = _make_hitl_decision({"name": "", "args": {}}, console)
        assert result == {"type": "approve"}

    def test_shell_piped_command_allowed(self, console: Console) -> None:
        """Piped shell commands where all segments are allowed should pass."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls", "grep"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "ls | grep test"}}, console
            )
            assert result == {"type": "approve"}

    def test_shell_piped_command_with_disallowed_segment(
        self, console: Console
    ) -> None:
        """Piped commands with a disallowed segment should be rejected."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "ls | rm file"}}, console
            )
            assert result["type"] == "reject"

    def test_shell_dangerous_pattern_rejected(self, console: Console) -> None:
        """Dangerous patterns rejected even if base command is allowed."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls"]
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "ls $(whoami)"}}, console
            )
            assert result["type"] == "reject"

    def test_shell_with_allow_all_approved(self, console: Console) -> None:
        """Shell commands should be approved when SHELL_ALLOW_ALL is set."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = SHELL_ALLOW_ALL
            result = _make_hitl_decision(
                {"name": "execute", "args": {"command": "rm -rf /"}}, console
            )
            assert result == {"type": "approve"}

    @pytest.mark.parametrize("tool_name", ["bash", "shell", "execute"])
    def test_all_shell_tool_names_recognised(
        self, tool_name: str, console: Console
    ) -> None:
        """All SHELL_TOOL_NAMES variants should be gated by the allow-list."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.shell_allow_list = ["ls"]
            result = _make_hitl_decision(
                {"name": tool_name, "args": {"command": "rm -rf /"}}, console
            )
            assert result["type"] == "reject"

    def test_collect_action_request_warnings_for_hidden_unicode(self) -> None:
        """Hidden Unicode in action args should generate warnings."""
        warnings = _collect_action_request_warnings(
            {"name": "execute", "args": {"command": "echo he\u200bllo"}}
        )
        assert warnings
        assert any("hidden Unicode" in warning for warning in warnings)

    def test_collect_action_request_warnings_for_suspicious_url(self) -> None:
        """Suspicious URLs in action args should generate warnings."""
        warnings = _collect_action_request_warnings(
            {"name": "fetch_url", "args": {"url": "https://аpple.com"}}
        )
        assert warnings
        assert any("URL warning" in warning for warning in warnings)

    def test_collect_action_request_warnings_nested_values(self) -> None:
        """Nested string values should be inspected recursively."""
        warnings = _collect_action_request_warnings(
            {
                "name": "fetch_url",
                "args": {"headers": {"Referer": "echo \u200bhello"}},
            }
        )
        assert warnings
        assert any("hidden Unicode" in warning for warning in warnings)


class TestBuildNonInteractiveHeader:
    """Tests for _build_non_interactive_header()."""

    def test_includes_agent_id(self) -> None:
        """Header should contain the agent identifier."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            header = _build_non_interactive_header("my-agent", "abc123")
        assert "Agent: my-agent" in header.plain
        # Non-default agent should not have "(default)" label
        assert "(default)" not in header.plain

    def test_default_agent_label(self) -> None:
        """Header should show '(default)' for the default agent name."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            header = _build_non_interactive_header("agent", "abc123")
        assert "Agent: agent (default)" in header.plain

    def test_includes_model_name(self) -> None:
        """Header should display model name when available."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = "gpt-5"
            header = _build_non_interactive_header("agent", "abc123")
        assert "Model: gpt-5" in header.plain

    def test_omits_model_when_none(self) -> None:
        """Header should not include model section when model_name is None."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            header = _build_non_interactive_header("agent", "abc123")
        assert "Model:" not in header.plain

    def test_includes_thread_id(self) -> None:
        """Header should contain the thread ID."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            header = _build_non_interactive_header("agent", "deadbeef")
        assert "Thread: deadbeef" in header.plain

    def test_thread_clickable_when_url_available(self) -> None:
        """Thread ID should be a hyperlink when LangSmith URL is available."""
        url = "https://smith.langchain.com/o/org/projects/p/proj/t/abc123"
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            with patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=url,
            ):
                header = _build_non_interactive_header(
                    "agent",
                    "abc123",
                    include_thread_link=True,
                )
        # Find the span containing the thread ID and verify it has a link
        for start, end, style in header._spans:
            text = header.plain[start:end]
            if text == "abc123" and isinstance(style, Style) and style.link:
                assert style.link == url
                break
        else:
            pytest.fail("Thread ID span with hyperlink not found")

    def test_default_header_does_not_lookup_langsmith(self) -> None:
        """Header should skip LangSmith lookup unless explicitly enabled."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            with patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
            ) as mock_build_url:
                _build_non_interactive_header("agent", "abc123")

        mock_build_url.assert_not_called()


class TestSandboxTypeForwarding:
    """Test that sandbox_type is forwarded to start_server_and_get_agent."""

    async def test_sandbox_type_passed_to_server(self) -> None:
        """run_non_interactive should forward sandbox_type to the server."""
        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ) as mock_start_server,
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(
                message="test task",
                sandbox_type="modal",
            )

        _, kwargs = mock_start_server.call_args
        assert kwargs["sandbox_type"] == "modal"


class TestQuietMode:
    """Tests for --quiet flag in run_non_interactive."""

    @pytest.mark.parametrize(
        ("quiet", "expected_kwargs"),
        [
            pytest.param(True, {"stderr": True}, id="quiet-redirects-to-stderr"),
            pytest.param(False, {}, id="default-uses-stdout"),
        ],
    )
    async def test_console_creation(
        self, quiet: bool, expected_kwargs: dict[str, object]
    ) -> None:
        """Console should use stderr when quiet=True, stdout otherwise."""
        mock_console = MagicMock(spec=Console)
        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.Console",
                return_value=mock_console,
            ) as mock_console_cls,
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="test", quiet=quiet)

        mock_console_cls.assert_called_once_with(**expected_kwargs)

    async def test_quiet_stdout_contains_only_agent_text(self) -> None:
        """In quiet mode, stdout should have only agent text."""
        # Build a fake AI message with a text block followed by a tool-call block
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.content_blocks = [
            {"type": "text", "text": "Hello from agent"},
            {"type": "tool_call_chunk", "name": "read_file", "id": "tc1", "index": 0},
        ]
        stream_chunks = [
            # 3-tuple: (namespace, stream_mode, data)
            ("", "messages", (ai_msg, {})),
        ]

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter(stream_chunks))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
            patch.object(sys, "stdout", stdout_buf),
            patch.object(sys, "stderr", stderr_buf),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="test", quiet=True)

        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        # Agent response text goes to stdout
        assert "Hello from agent" in stdout
        # Diagnostic messages should NOT be on stdout
        assert "Calling tool" not in stdout
        assert "Task completed" not in stdout
        assert "Running task" not in stdout
        # Tool notifications still go to stderr
        assert "Calling tool" in stderr or "read_file" in stderr
        # Header and completion messages are fully suppressed in quiet mode
        assert "Task completed" not in stderr
        assert "Running task" not in stderr


class TestNoStreamMode:
    """Tests for --no-stream flag in run_non_interactive."""

    async def test_no_stream_buffers_output(self) -> None:
        """In no-stream mode, stdout should receive text only after completion."""
        # Build two text chunks to verify buffering vs streaming
        ai_msg1 = MagicMock(spec=AIMessage)
        ai_msg1.content_blocks = [{"type": "text", "text": "Hello "}]
        ai_msg2 = MagicMock(spec=AIMessage)
        ai_msg2.content_blocks = [{"type": "text", "text": "world"}]

        stream_chunks = [
            ("", "messages", (ai_msg1, {})),
            ("", "messages", (ai_msg2, {})),
        ]

        stdout_writes: list[str] = []

        class TrackingStringIO(io.StringIO):
            """StringIO that records each write call separately."""

            def write(self, s: str) -> int:
                stdout_writes.append(s)
                return super().write(s)

        stdout_buf = TrackingStringIO()

        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter(stream_chunks))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
            patch.object(sys, "stdout", stdout_buf),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="test", quiet=True, stream=False)

        stdout = stdout_buf.getvalue()
        assert "Hello world" in stdout

        # Verify the text was NOT written incrementally — the first
        # text write should contain the full concatenated response
        text_writes = [w for w in stdout_writes if w != "\n"]
        assert len(text_writes) == 1
        assert text_writes[0] == "Hello world"

    async def test_stream_mode_writes_incrementally(self) -> None:
        """Default stream mode should write text chunks as they arrive."""
        ai_msg1 = MagicMock(spec=AIMessage)
        ai_msg1.content_blocks = [{"type": "text", "text": "Hello "}]
        ai_msg2 = MagicMock(spec=AIMessage)
        ai_msg2.content_blocks = [{"type": "text", "text": "world"}]

        stream_chunks = [
            ("", "messages", (ai_msg1, {})),
            ("", "messages", (ai_msg2, {})),
        ]

        stdout_writes: list[str] = []

        class TrackingStringIO(io.StringIO):
            """StringIO that records each write call separately."""

            def write(self, s: str) -> int:
                stdout_writes.append(s)
                return super().write(s)

        stdout_buf = TrackingStringIO()

        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter(stream_chunks))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
            patch.object(sys, "stdout", stdout_buf),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="test", quiet=True, stream=True)

        stdout = stdout_buf.getvalue()
        assert "Hello world" in stdout

        # Verify text was written incrementally (two separate writes)
        text_writes = [w for w in stdout_writes if w != "\n"]
        assert len(text_writes) == 2
        assert text_writes[0] == "Hello "
        assert text_writes[1] == "world"


class TestFastFollowLangsmithLink:
    """Tests for best-effort fast-follow LangSmith link output."""

    async def test_prints_link_when_lookup_ready(self) -> None:
        """Should print LangSmith link before completion when ready."""
        mock_console = MagicMock(spec=Console)
        ready_state = ThreadUrlLookupState()
        ready_state.done.set()
        ready_state.url = (
            "https://smith.langchain.com/o/org/projects/p/proj/t/test-thread"
        )

        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.Console",
                return_value=mock_console,
            ),
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive._start_langsmith_thread_url_lookup",
                return_value=ready_state,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="test", quiet=False)

        printed = [
            str(call.args[0]) for call in mock_console.print.call_args_list if call.args
        ]
        assert any("View in LangSmith:" in line for line in printed)

    async def test_skips_link_when_lookup_not_ready(self) -> None:
        """Should not wait for or print link when lookup is still in flight."""
        mock_console = MagicMock(spec=Console)
        pending_state = ThreadUrlLookupState()
        pending_state.url = (
            "https://smith.langchain.com/o/org/projects/p/proj/t/test-thread"
        )

        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.Console",
                return_value=mock_console,
            ),
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive._start_langsmith_thread_url_lookup",
                return_value=pending_state,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="test", quiet=False)

        printed = [
            str(call.args[0]) for call in mock_console.print.call_args_list if call.args
        ]
        assert not any("View in LangSmith:" in line for line in printed)

    async def test_skips_link_when_lookup_done_but_url_none(self) -> None:
        """Should not print link when lookup completed but URL is None."""
        mock_console = MagicMock(spec=Console)
        done_no_url = ThreadUrlLookupState()
        done_no_url.done.set()

        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.Console",
                return_value=mock_console,
            ),
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive._start_langsmith_thread_url_lookup",
                return_value=done_no_url,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="test", quiet=False)

        printed = [
            str(call.args[0]) for call in mock_console.print.call_args_list if call.args
        ]
        assert not any("View in LangSmith:" in line for line in printed)

    async def test_quiet_mode_skips_thread_url_lookup(self) -> None:
        """Should not start LangSmith URL lookup when quiet=True."""
        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.Console",
                return_value=MagicMock(spec=Console),
            ),
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive._start_langsmith_thread_url_lookup",
            ) as mock_lookup,
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="test", quiet=True)

        mock_lookup.assert_not_called()


class TestStartLangsmithThreadUrlLookup:
    """Tests for _start_langsmith_thread_url_lookup."""

    def test_sets_url_on_success(self) -> None:
        """Should populate state.url when build succeeds."""
        url = "https://smith.langchain.com/o/org/projects/p/proj/t/tid"
        with patch(
            "deepagents_cli.non_interactive.build_langsmith_thread_url",
            return_value=url,
        ):
            state = _start_langsmith_thread_url_lookup("tid")
            assert state.done.wait(timeout=2.0)
        assert state.url == url

    def test_signals_done_on_exception(self) -> None:
        """Should signal done and leave url as None when build raises."""
        with patch(
            "deepagents_cli.non_interactive.build_langsmith_thread_url",
            side_effect=RuntimeError("boom"),
        ):
            state = _start_langsmith_thread_url_lookup("tid")
            assert state.done.wait(timeout=2.0)
        assert state.url is None

    def test_signals_done_when_url_is_none(self) -> None:
        """Should signal done when build returns None."""
        with patch(
            "deepagents_cli.non_interactive.build_langsmith_thread_url",
            return_value=None,
        ):
            state = _start_langsmith_thread_url_lookup("tid")
            assert state.done.wait(timeout=2.0)
        assert state.url is None


class TestShellAllowListDecisionLogic:
    """Tests for shell allow-list → auto_approve / interrupt_shell_only."""

    @pytest.mark.parametrize(
        (
            "shell_allow_list",
            "expected_auto",
            "expected_shell_only",
            "expected_allow_list",
        ),
        [
            pytest.param(
                None,
                True,
                False,
                None,
                id="no-allow-list-auto-approves",
            ),
            pytest.param(
                ["ls", "cat"],
                False,
                True,
                ["ls", "cat"],
                id="restrictive-list-interrupts-shell-only",
            ),
            pytest.param(
                SHELL_ALLOW_ALL,
                True,
                False,
                None,
                id="allow-all-auto-approves",
            ),
        ],
    )
    async def test_shell_auto_approve_branches(
        self,
        shell_allow_list: list[str] | None,
        expected_auto: bool,
        expected_shell_only: bool,
        expected_allow_list: list[str] | None,
    ) -> None:
        """Verify start_server_and_get_agent receives correct flags."""
        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ) as mock_start_server,
        ):
            mock_settings.shell_allow_list = shell_allow_list
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="test task")

        _, kwargs = mock_start_server.call_args
        assert kwargs["auto_approve"] is expected_auto
        assert kwargs["interrupt_shell_only"] is expected_shell_only
        assert kwargs["shell_allow_list"] == expected_allow_list


class TestNonInteractivePrompt:
    """Tests that run_non_interactive passes interactive=False."""

    async def test_passes_interactive_false(self) -> None:
        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ) as mock_start_server,
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="do the thing")

        _, kwargs = mock_start_server.call_args
        assert kwargs["interactive"] is False

    async def test_initial_skill_wraps_prompt_and_metadata(self) -> None:
        """Headless skill execution should send wrapped prompt + `__skill`."""
        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()
        skill = {
            "name": "code-review",
            "description": "Review code changes",
            "path": "/skills/code-review/SKILL.md",
            "license": None,
            "compatibility": None,
            "metadata": {},
            "allowed_tools": [],
            "source": "user",
        }

        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.skills.invocation.discover_skills_and_roots",
                return_value=([skill], []),
            ),
            patch(
                "deepagents_cli.skills.load.load_skill_content",
                return_value="# Instructions\nDo stuff",
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(
                message="review this patch",
                initial_skill="code-review",
                quiet=True,
            )

        stream_input = mock_agent.astream.call_args.args[0]
        user_msg = stream_input["messages"][0]
        assert "I'm invoking the skill `code-review`." in user_msg["content"]
        assert "**User request:** review this patch" in user_msg["content"]
        assert user_msg["additional_kwargs"]["__skill"]["name"] == "code-review"
        assert user_msg["additional_kwargs"]["__skill"]["args"] == "review this patch"

    async def test_initial_skill_missing_returns_error_without_starting_server(
        self,
    ) -> None:
        """Missing headless skill should fail before the server starts."""
        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.settings",
            ) as mock_settings,
            patch(
                "deepagents_cli.skills.invocation.discover_skills_and_roots",
                return_value=([], []),
            ),
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
            ) as mock_start_server,
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            result = await run_non_interactive(
                message="review this patch",
                initial_skill="missing-skill",
                quiet=True,
            )

        assert result == 1
        mock_start_server.assert_not_awaited()


def _make_interrupt_chunk(interrupt_id: str = "i1") -> tuple:
    """Return a stream chunk that triggers one HITL interrupt.

    The interrupt value is a dict that would normally need to pass the
    HITLRequest Pydantic validator. Tests that call this helper must also
    patch `_HITL_REQUEST_ADAPTER.validate_python` to pass-through so that
    validation is bypassed and `state.interrupt_occurred` is set correctly.
    """
    interrupt = MagicMock()
    interrupt.id = interrupt_id
    interrupt.value = {
        "action_requests": [{"name": "read_file", "args": {"path": "/tmp/f"}}]
    }
    return ("", "updates", {"__interrupt__": [interrupt]})


def _make_looping_agent() -> MagicMock:
    """Return a mock agent whose astream always yields one interrupt chunk."""
    chunk = _make_interrupt_chunk()
    mock_agent = MagicMock()
    mock_agent.astream = MagicMock(side_effect=lambda *_, **__: _async_iter([chunk]))
    return mock_agent


class TestMaxTurns:
    """Tests for max_turns parameter in _run_agent_loop."""

    async def test_raises_after_user_limit(self) -> None:
        """HITLIterationLimitError is raised after max_turns HITL iterations."""
        agent = _make_looping_agent()
        console = Console(quiet=True)
        file_op_tracker = MagicMock()
        file_op_tracker.complete_with_message.return_value = None
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        with (
            patch(
                "deepagents_cli.non_interactive.dispatch_hook", new_callable=AsyncMock
            ),
            patch("deepagents_cli.non_interactive.dispatch_hook_fire_and_forget"),
            patch("deepagents_cli.non_interactive.settings") as mock_settings,
            patch(
                "deepagents_cli.non_interactive._HITL_REQUEST_ADAPTER"
            ) as mock_adapter,
        ):
            mock_settings.shell_allow_list = None
            mock_settings.model_name = ""
            mock_adapter.validate_python.side_effect = lambda v: v
            with pytest.raises(HITLIterationLimitError) as exc_info:
                await _run_agent_loop(
                    agent,
                    "task",
                    config,
                    console,
                    file_op_tracker,
                    quiet=True,
                    max_turns=1,
                )
        assert "--max-turns 1" in str(exc_info.value)

    async def test_error_message_names_user_flag(self) -> None:
        """Error message references --max-turns and tells the user how to fix it."""
        agent = _make_looping_agent()
        console = Console(quiet=True)
        file_op_tracker = MagicMock()
        file_op_tracker.complete_with_message.return_value = None
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        with (
            patch(
                "deepagents_cli.non_interactive.dispatch_hook", new_callable=AsyncMock
            ),
            patch("deepagents_cli.non_interactive.dispatch_hook_fire_and_forget"),
            patch("deepagents_cli.non_interactive.settings") as mock_settings,
            patch(
                "deepagents_cli.non_interactive._HITL_REQUEST_ADAPTER"
            ) as mock_adapter,
        ):
            mock_settings.shell_allow_list = None
            mock_settings.model_name = ""
            mock_adapter.validate_python.side_effect = lambda v: v
            with pytest.raises(HITLIterationLimitError) as exc_info:
                await _run_agent_loop(
                    agent,
                    "task",
                    config,
                    console,
                    file_op_tracker,
                    quiet=True,
                    max_turns=2,
                )
        msg = str(exc_info.value)
        assert "--max-turns 2" in msg
        assert "Increase --max-turns" in msg

    async def test_max_turns_above_default_is_honoured(self) -> None:
        """User's --max-turns overrides the internal safety default (no clamp)."""
        # Patch the default ceiling to 2 — if the old clamp logic were still
        # in place, max_turns=4 would be truncated to 2 and the agent would
        # only run 3 astream calls before raising. Without clamping, the
        # loop must run 5 astream calls (1 initial + 4 HITL rounds).
        agent = _make_looping_agent()
        console = Console(quiet=True)
        file_op_tracker = MagicMock()
        file_op_tracker.complete_with_message.return_value = None
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        with (
            patch(
                "deepagents_cli.non_interactive.dispatch_hook", new_callable=AsyncMock
            ),
            patch("deepagents_cli.non_interactive.dispatch_hook_fire_and_forget"),
            patch("deepagents_cli.non_interactive.settings") as mock_settings,
            patch(
                "deepagents_cli.non_interactive._HITL_REQUEST_ADAPTER"
            ) as mock_adapter,
            patch("deepagents_cli.non_interactive._MAX_HITL_ITERATIONS", 2),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.model_name = ""
            mock_adapter.validate_python.side_effect = lambda v: v
            with pytest.raises(HITLIterationLimitError) as exc_info:
                await _run_agent_loop(
                    agent,
                    "task",
                    config,
                    console,
                    file_op_tracker,
                    quiet=True,
                    max_turns=4,
                )
        msg = str(exc_info.value)
        assert "Exceeded 4 agentic turns" in msg
        assert "--max-turns 4" in msg
        assert "internal safety default" not in msg
        assert agent.astream.call_count == 5

    async def test_no_max_turns_uses_internal_default(self) -> None:
        """Omitting max_turns falls back to the internal safety default."""
        agent = _make_looping_agent()
        console = Console(quiet=True)
        file_op_tracker = MagicMock()
        file_op_tracker.complete_with_message.return_value = None
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        with (
            patch(
                "deepagents_cli.non_interactive.dispatch_hook", new_callable=AsyncMock
            ),
            patch("deepagents_cli.non_interactive.dispatch_hook_fire_and_forget"),
            patch("deepagents_cli.non_interactive.settings") as mock_settings,
            patch(
                "deepagents_cli.non_interactive._HITL_REQUEST_ADAPTER"
            ) as mock_adapter,
            patch("deepagents_cli.non_interactive._MAX_HITL_ITERATIONS", 1),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.model_name = ""
            mock_adapter.validate_python.side_effect = lambda v: v
            with pytest.raises(HITLIterationLimitError) as exc_info:
                await _run_agent_loop(
                    agent,
                    "task",
                    config,
                    console,
                    file_op_tracker,
                    quiet=True,
                    max_turns=None,
                )
        msg = str(exc_info.value)
        assert "internal safety default of 1" in msg

    async def test_max_turns_forwarded_from_run_non_interactive(self) -> None:
        """run_non_interactive passes max_turns through to _run_agent_loop."""
        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch("deepagents_cli.non_interactive.settings") as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.non_interactive._run_agent_loop",
                new_callable=AsyncMock,
            ) as mock_loop,
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="task", max_turns=7)

        _, kwargs = mock_loop.call_args
        assert kwargs.get("max_turns") == 7

    async def test_max_turns_none_forwarded_by_default(self) -> None:
        """run_non_interactive forwards max_turns=None when not supplied."""
        mock_agent = MagicMock()
        mock_agent.astream = MagicMock(return_value=_async_iter([]))
        mock_server_proc = MagicMock()

        with (
            patch(
                "deepagents_cli.non_interactive.create_model",
                return_value=ModelResult(
                    model=MagicMock(),
                    model_name="test-model",
                    provider="test",
                ),
            ),
            patch(
                "deepagents_cli.non_interactive.generate_thread_id",
                return_value="test-thread",
            ),
            patch("deepagents_cli.non_interactive.settings") as mock_settings,
            patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ),
            patch(
                "deepagents_cli.non_interactive._run_agent_loop",
                new_callable=AsyncMock,
            ) as mock_loop,
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(mock_agent, mock_server_proc, None),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            await run_non_interactive(message="task")

        _, kwargs = mock_loop.call_args
        assert kwargs.get("max_turns") is None

    async def test_honours_full_user_budget_before_raising(self) -> None:
        """With max_turns=N, N full HITL rounds run before the guard trips.

        Pins the off-by-one semantics: `iterations > effective_limit` (strict
        >) must allow exactly N HITL iterations on top of the initial stream.
        If the guard flipped to `>=`, call_count would be N instead of N+1.
        """
        agent = _make_looping_agent()
        console = Console(quiet=True)
        file_op_tracker = MagicMock()
        file_op_tracker.complete_with_message.return_value = None
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        with (
            patch(
                "deepagents_cli.non_interactive.dispatch_hook", new_callable=AsyncMock
            ),
            patch("deepagents_cli.non_interactive.dispatch_hook_fire_and_forget"),
            patch("deepagents_cli.non_interactive.settings") as mock_settings,
            patch(
                "deepagents_cli.non_interactive._HITL_REQUEST_ADAPTER"
            ) as mock_adapter,
        ):
            mock_settings.shell_allow_list = None
            mock_settings.model_name = ""
            mock_adapter.validate_python.side_effect = lambda v: v
            with pytest.raises(HITLIterationLimitError):
                await _run_agent_loop(
                    agent,
                    "task",
                    config,
                    console,
                    file_op_tracker,
                    quiet=True,
                    max_turns=3,
                )
        assert agent.astream.call_count == 4  # 1 initial + 3 HITL rounds


async def _async_iter(items: list[object]) -> AsyncIterator[object]:  # noqa: RUF029
    """Create an async iterator from a list for testing."""
    for item in items:
        yield item
