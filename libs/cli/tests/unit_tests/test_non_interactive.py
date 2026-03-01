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
    _build_non_interactive_header,
    _make_hitl_decision,
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


class TestBuildNonInteractiveHeader:
    """Tests for _build_non_interactive_header()."""

    def test_includes_agent_id(self) -> None:
        """Header should contain the agent identifier."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            with patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ):
                header = _build_non_interactive_header("my-agent", "abc123")
        assert "Agent: my-agent" in header.plain
        # Non-default agent should not have "(default)" label
        assert "(default)" not in header.plain

    def test_default_agent_label(self) -> None:
        """Header should show '(default)' for the default agent name."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            with patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ):
                header = _build_non_interactive_header("agent", "abc123")
        assert "Agent: agent (default)" in header.plain

    def test_includes_model_name(self) -> None:
        """Header should display model name when available."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = "gpt-5"
            with patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ):
                header = _build_non_interactive_header("agent", "abc123")
        assert "Model: gpt-5" in header.plain

    def test_omits_model_when_none(self) -> None:
        """Header should not include model section when model_name is None."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            with patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ):
                header = _build_non_interactive_header("agent", "abc123")
        assert "Model:" not in header.plain

    def test_includes_thread_id(self) -> None:
        """Header should contain the thread ID."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            with patch(
                "deepagents_cli.non_interactive.build_langsmith_thread_url",
                return_value=None,
            ):
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
                header = _build_non_interactive_header("agent", "abc123")
        # Find the span containing the thread ID and verify it has a link
        for start, end, style in header._spans:
            text = header.plain[start:end]
            if text == "abc123" and isinstance(style, Style) and style.link:
                assert style.link == url
                break
        else:
            pytest.fail("Thread ID span with hyperlink not found")


class TestSandboxSetupForwarding:
    """Test that sandbox_setup is forwarded to create_sandbox."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


async def _async_iter(items: list[object]) -> AsyncIterator[object]:  # noqa: RUF029
    """Create an async iterator from a list for testing."""
    for item in items:
        yield item
