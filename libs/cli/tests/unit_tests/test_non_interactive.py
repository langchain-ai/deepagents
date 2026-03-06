"""Tests for non-interactive mode HITL decision logic."""

import io
import sys
from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console
from rich.style import Style
from rich.text import Text

from deepagents_cli.config import ModelResult
from deepagents_cli.non_interactive import (
    ThreadUrlLookupState,
    _build_non_interactive_header,
    _collect_action_request_warnings,
    _make_hitl_decision,
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
                "name": "http_request",
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


class TestSandboxSetupForwarding:
    """Test that sandbox_setup is forwarded to create_sandbox."""

    async def test_sandbox_setup_passed_to_create_sandbox(self) -> None:
        """run_non_interactive should forward sandbox_setup to create_sandbox.

        When both --sandbox and --sandbox-setup are provided, the
        setup_script_path must reach create_sandbox so the setup script
        actually runs inside the sandbox.
        """
        mock_backend = MagicMock()
        mock_backend.id = "sandbox-123"

        # Capture kwargs passed to create_sandbox
        captured_kwargs: list[dict[str, object]] = []

        @contextmanager
        def fake_create_sandbox(
            _provider: str,
            *,
            sandbox_id: str | None = None,  # noqa: ARG001 # match create_sandbox signature
            setup_script_path: str | None = None,
        ) -> Generator[MagicMock, None, None]:
            captured_kwargs.append({"setup_script_path": setup_script_path})
            yield mock_backend

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
                "deepagents_cli.integrations.sandbox_factory.create_sandbox",
                side_effect=fake_create_sandbox,
            ),
            patch(
                "deepagents_cli.non_interactive.get_checkpointer",
            ) as mock_checkpointer,
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            # Make the checkpointer async context manager return a mock
            mock_cp = MagicMock()
            mock_checkpointer.return_value.__aenter__ = MagicMock(return_value=mock_cp)
            mock_checkpointer.return_value.__aexit__ = MagicMock(return_value=None)

            # Make create_cli_agent return a mock agent that immediately finishes
            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter([]))
            mock_create_agent.return_value = (mock_agent, MagicMock())

            await run_non_interactive(
                message="test task",
                sandbox_type="modal",
                sandbox_setup="/path/to/setup.sh",
            )

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["setup_script_path"] == "/path/to/setup.sh"


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
                "deepagents_cli.non_interactive.get_checkpointer",
            ) as mock_checkpointer,
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            mock_cp = MagicMock()
            mock_checkpointer.return_value.__aenter__ = MagicMock(return_value=mock_cp)
            mock_checkpointer.return_value.__aexit__ = MagicMock(return_value=None)

            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter([]))
            mock_create_agent.return_value = (mock_agent, MagicMock())

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

        mock_cp = MagicMock()
        mock_checkpointer_cm = AsyncMock()
        mock_checkpointer_cm.__aenter__.return_value = mock_cp
        mock_checkpointer_cm.__aexit__.return_value = None

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
                "deepagents_cli.non_interactive.get_checkpointer",
                return_value=mock_checkpointer_cm,
            ),
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
            patch.object(sys, "stdout", stdout_buf),
            patch.object(sys, "stderr", stderr_buf),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter(stream_chunks))
            mock_create_agent.return_value = (mock_agent, MagicMock())

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

        mock_cp = MagicMock()
        mock_checkpointer_cm = AsyncMock()
        mock_checkpointer_cm.__aenter__.return_value = mock_cp
        mock_checkpointer_cm.__aexit__.return_value = None

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
                "deepagents_cli.non_interactive.get_checkpointer",
                return_value=mock_checkpointer_cm,
            ),
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
            patch.object(sys, "stdout", stdout_buf),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter(stream_chunks))
            mock_create_agent.return_value = (mock_agent, MagicMock())

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

        mock_cp = MagicMock()
        mock_checkpointer_cm = AsyncMock()
        mock_checkpointer_cm.__aenter__.return_value = mock_cp
        mock_checkpointer_cm.__aexit__.return_value = None

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
                "deepagents_cli.non_interactive.get_checkpointer",
                return_value=mock_checkpointer_cm,
            ),
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
            patch.object(sys, "stdout", stdout_buf),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter(stream_chunks))
            mock_create_agent.return_value = (mock_agent, MagicMock())

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
                "deepagents_cli.non_interactive.get_checkpointer",
            ) as mock_checkpointer,
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            mock_cp = MagicMock()
            mock_checkpointer_cm = AsyncMock()
            mock_checkpointer_cm.__aenter__.return_value = mock_cp
            mock_checkpointer_cm.__aexit__.return_value = None
            mock_checkpointer.return_value = mock_checkpointer_cm

            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter([]))
            mock_create_agent.return_value = (mock_agent, MagicMock())

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
                "deepagents_cli.non_interactive.get_checkpointer",
            ) as mock_checkpointer,
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            mock_cp = MagicMock()
            mock_checkpointer_cm = AsyncMock()
            mock_checkpointer_cm.__aenter__.return_value = mock_cp
            mock_checkpointer_cm.__aexit__.return_value = None
            mock_checkpointer.return_value = mock_checkpointer_cm

            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter([]))
            mock_create_agent.return_value = (mock_agent, MagicMock())

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
                "deepagents_cli.non_interactive.get_checkpointer",
            ) as mock_checkpointer,
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            mock_cp = MagicMock()
            mock_checkpointer_cm = AsyncMock()
            mock_checkpointer_cm.__aenter__.return_value = mock_cp
            mock_checkpointer_cm.__aexit__.return_value = None
            mock_checkpointer.return_value = mock_checkpointer_cm

            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter([]))
            mock_create_agent.return_value = (mock_agent, MagicMock())

            await run_non_interactive(message="test", quiet=False)

        printed = [
            str(call.args[0]) for call in mock_console.print.call_args_list if call.args
        ]
        assert not any("View in LangSmith:" in line for line in printed)

    async def test_quiet_mode_skips_thread_url_lookup(self) -> None:
        """Should not start LangSmith URL lookup when quiet=True."""
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
                "deepagents_cli.non_interactive.get_checkpointer",
            ) as mock_checkpointer,
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
        ):
            mock_settings.shell_allow_list = None
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            mock_cp = MagicMock()
            mock_checkpointer_cm = AsyncMock()
            mock_checkpointer_cm.__aenter__.return_value = mock_cp
            mock_checkpointer_cm.__aexit__.return_value = None
            mock_checkpointer.return_value = mock_checkpointer_cm

            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter([]))
            mock_create_agent.return_value = (mock_agent, MagicMock())

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


class TestAutoApproveShellLogic:
    """Tests for the 3-branch auto_approve + shell_allow_list decision."""

    @pytest.mark.parametrize(
        ("auto_approve", "shell_allow_list", "expected_enable_shell", "expected_auto"),
        [
            pytest.param(
                True,
                None,
                True,
                True,
                id="auto-approve-enables-shell-and-auto-approve",
            ),
            pytest.param(
                True,
                ["ls", "cat"],
                True,
                True,
                id="auto-approve-overrides-allow-list",
            ),
            pytest.param(
                False,
                ["ls", "cat"],
                True,
                False,
                id="allow-list-enables-shell-without-auto-approve",
            ),
            pytest.param(
                False,
                None,
                False,
                True,
                id="no-flags-disables-shell-with-auto-approve",
            ),
        ],
    )
    async def test_shell_auto_approve_branches(
        self,
        auto_approve: bool,
        shell_allow_list: list[str] | None,
        expected_enable_shell: bool,
        expected_auto: bool,
    ) -> None:
        """Verify create_cli_agent receives correct enable_shell and auto_approve."""
        mock_cp = MagicMock()
        mock_checkpointer_cm = AsyncMock()
        mock_checkpointer_cm.__aenter__.return_value = mock_cp
        mock_checkpointer_cm.__aexit__.return_value = None

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
                "deepagents_cli.non_interactive.get_checkpointer",
                return_value=mock_checkpointer_cm,
            ),
            patch(
                "deepagents_cli.non_interactive.create_cli_agent",
            ) as mock_create_agent,
            patch(
                "deepagents_cli.mcp_tools.resolve_and_load_mcp_tools",
                return_value=([], None, []),
            ),
        ):
            mock_settings.shell_allow_list = shell_allow_list
            mock_settings.has_tavily = False
            mock_settings.model_name = None

            mock_agent = MagicMock()
            mock_agent.astream = MagicMock(return_value=_async_iter([]))
            mock_create_agent.return_value = (mock_agent, MagicMock())

            await run_non_interactive(
                message="test task",
                auto_approve=auto_approve,
            )

        _, kwargs = mock_create_agent.call_args
        assert kwargs["enable_shell"] is expected_enable_shell
        assert kwargs["auto_approve"] is expected_auto


async def _async_iter(items: list[object]) -> AsyncIterator[object]:  # noqa: RUF029
    """Create an async iterator from a list for testing."""
    for item in items:
        yield item
