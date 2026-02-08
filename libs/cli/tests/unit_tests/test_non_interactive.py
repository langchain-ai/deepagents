"""Tests for non-interactive mode HITL decision logic."""

from unittest.mock import patch

import pytest
from rich.console import Console
from rich.style import Style
from rich.text import Text

from deepagents_cli.non_interactive import (
    _build_non_interactive_header,
    _get_thread_url,
    _make_hitl_decision,
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
                "deepagents_cli.non_interactive._get_thread_url", return_value=None
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
                "deepagents_cli.non_interactive._get_thread_url", return_value=None
            ):
                header = _build_non_interactive_header("agent", "abc123")
        assert "Agent: agent (default)" in header.plain

    def test_includes_model_name(self) -> None:
        """Header should display model name when available."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = "gpt-5"
            with patch(
                "deepagents_cli.non_interactive._get_thread_url", return_value=None
            ):
                header = _build_non_interactive_header("agent", "abc123")
        assert "Model: gpt-5" in header.plain

    def test_omits_model_when_none(self) -> None:
        """Header should not include model section when model_name is None."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            with patch(
                "deepagents_cli.non_interactive._get_thread_url", return_value=None
            ):
                header = _build_non_interactive_header("agent", "abc123")
        assert "Model:" not in header.plain

    def test_includes_thread_id(self) -> None:
        """Header should contain the thread ID."""
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            with patch(
                "deepagents_cli.non_interactive._get_thread_url", return_value=None
            ):
                header = _build_non_interactive_header("agent", "deadbeef")
        assert "Thread: deadbeef" in header.plain

    def test_thread_clickable_when_url_available(self) -> None:
        """Thread ID should be a hyperlink when LangSmith URL is available."""
        url = "https://smith.langchain.com/o/org/projects/p/proj/t/abc123"
        with patch("deepagents_cli.non_interactive.settings") as mock_settings:
            mock_settings.model_name = None
            with patch(
                "deepagents_cli.non_interactive._get_thread_url", return_value=url
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


class TestGetThreadUrl:
    """Tests for _get_thread_url().

    The function delegates to ``get_langsmith_project_name`` and
    ``fetch_langsmith_project_url`` from config, so we mock those
    rather than env vars / the LangSmith client directly.
    """

    def test_returns_none_when_langsmith_not_configured(self) -> None:
        """Should return None when get_langsmith_project_name returns None."""
        with patch(
            "deepagents_cli.non_interactive.get_langsmith_project_name",
            return_value=None,
        ):
            assert _get_thread_url("abc123") is None

    def test_returns_none_when_project_url_unavailable(self) -> None:
        """Should return None when fetch_langsmith_project_url returns None."""
        with (
            patch(
                "deepagents_cli.non_interactive.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch(
                "deepagents_cli.non_interactive.fetch_langsmith_project_url",
                return_value=None,
            ),
        ):
            assert _get_thread_url("abc123") is None

    def test_returns_url_when_configured(self) -> None:
        """Should return a full thread URL when LangSmith is configured."""
        project_url = "https://smith.langchain.com/o/org/projects/p/proj"
        with (
            patch(
                "deepagents_cli.non_interactive.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch(
                "deepagents_cli.non_interactive.fetch_langsmith_project_url",
                return_value=project_url,
            ),
        ):
            result = _get_thread_url("thread42")

        assert result == f"{project_url}/t/thread42"

    def test_strips_trailing_slash_from_project_url(self) -> None:
        """Should strip trailing slash before appending thread path."""
        project_url = "https://smith.langchain.com/o/org/projects/p/proj/"
        with (
            patch(
                "deepagents_cli.non_interactive.get_langsmith_project_name",
                return_value="default",
            ),
            patch(
                "deepagents_cli.non_interactive.fetch_langsmith_project_url",
                return_value=project_url,
            ),
        ):
            result = _get_thread_url("abc")

        assert result == "https://smith.langchain.com/o/org/projects/p/proj/t/abc"
