"""Tests for CLI argument parsing."""

import io
import sys
from unittest.mock import patch

import pytest
from rich.console import Console

from deepagents_cli.agent import DEFAULT_AGENT_NAME
from deepagents_cli.main import _DEFAULT_AGENT_NAME, parse_args


class TestInitialPromptArg:
    """Tests for -m/--message initial prompt argument."""

    def test_short_flag(self) -> None:
        """Verify -m sets initial_prompt."""
        with patch.object(sys, "argv", ["deepagents", "-m", "hello world"]):
            args = parse_args()
        assert args.initial_prompt == "hello world"

    def test_long_flag(self) -> None:
        """Verify --message sets initial_prompt."""
        with patch.object(sys, "argv", ["deepagents", "--message", "hello world"]):
            args = parse_args()
        assert args.initial_prompt == "hello world"

    def test_no_flag(self) -> None:
        """Verify initial_prompt is None when not provided."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.initial_prompt is None

    def test_with_other_args(self) -> None:
        """Verify -m works alongside other arguments."""
        with patch.object(
            sys, "argv", ["deepagents", "--agent", "myagent", "-m", "do something"]
        ):
            args = parse_args()
        assert args.initial_prompt == "do something"
        assert args.agent == "myagent"

    def test_empty_string(self) -> None:
        """Verify empty string is accepted."""
        with patch.object(sys, "argv", ["deepagents", "-m", ""]):
            args = parse_args()
        assert args.initial_prompt == ""


class TestResumeArg:
    """Tests for -r/--resume thread resume argument."""

    def test_short_flag_no_value(self) -> None:
        """Verify -r without value sets resume_thread to __MOST_RECENT__."""
        with patch.object(sys, "argv", ["deepagents", "-r"]):
            args = parse_args()
        assert args.resume_thread == "__MOST_RECENT__"

    def test_short_flag_with_value(self) -> None:
        """Verify -r with ID sets resume_thread to that ID."""
        with patch.object(sys, "argv", ["deepagents", "-r", "abc12345"]):
            args = parse_args()
        assert args.resume_thread == "abc12345"

    def test_long_flag_no_value(self) -> None:
        """Verify --resume without value sets resume_thread to __MOST_RECENT__."""
        with patch.object(sys, "argv", ["deepagents", "--resume"]):
            args = parse_args()
        assert args.resume_thread == "__MOST_RECENT__"

    def test_long_flag_with_value(self) -> None:
        """Verify --resume with ID sets resume_thread to that ID."""
        with patch.object(sys, "argv", ["deepagents", "--resume", "xyz99999"]):
            args = parse_args()
        assert args.resume_thread == "xyz99999"

    def test_no_flag(self) -> None:
        """Verify resume_thread is None when not provided."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.resume_thread is None

    def test_with_other_args(self) -> None:
        """Verify -r works alongside --agent and -m."""
        with patch.object(
            sys, "argv", ["deepagents", "--agent", "myagent", "-r", "thread123"]
        ):
            args = parse_args()
        assert args.resume_thread == "thread123"
        assert args.agent == "myagent"

    def test_resume_with_message(self) -> None:
        """Verify -r works with -m initial message."""
        with patch.object(
            sys, "argv", ["deepagents", "-r", "thread456", "-m", "continue work"]
        ):
            args = parse_args()
        assert args.resume_thread == "thread456"
        assert args.initial_prompt == "continue work"


class TestTopLevelHelp:
    """Test that `deepagents -h` shows the global help screen via _make_help_action."""

    def test_top_level_help_exits_cleanly(self) -> None:
        """Running `deepagents -h` should show help and exit with code 0."""
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=120)

        with (
            patch.object(sys, "argv", ["deepagents", "-h"]),
            patch("deepagents_cli.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()

        assert exc_info.value.code in (0, None)
        output = buf.getvalue()

        # Should contain global help content
        assert "deepagents" in output.lower()
        assert "--help" in output

    def test_help_subcommand_parses(self) -> None:
        """Running `deepagents help` should parse as command='help'.

        The actual help display happens in `cli_main()`, not `parse_args()`.
        """
        with patch.object(sys, "argv", ["deepagents", "help"]):
            args = parse_args()
        assert args.command == "help"


class TestSubcommandHelpFlags:
    """Test that each subcommand's -h shows its own help screen (not global)."""

    def _run_help(
        self, argv: list[str], must_contain: str, must_not_contain: str
    ) -> None:
        """Run parse_args with *argv* and assert help output boundaries.

        Args:
            argv: sys.argv override.
            must_contain: Substring that must be present in the output.
            must_not_contain: Substring that must NOT be present.
        """
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=120)

        with (
            patch.object(sys, "argv", argv),
            patch("deepagents_cli.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()

        assert exc_info.value.code in (0, None)
        output = buf.getvalue()
        assert must_contain in output
        assert must_not_contain not in output

    def test_list_help(self) -> None:
        """Running `deepagents list -h` should show list-specific help."""
        self._run_help(
            ["deepagents", "list", "-h"],
            must_contain="List all agents",
            must_not_contain="--sandbox",
        )

    def test_reset_help(self) -> None:
        """Running `deepagents reset -h` should show reset-specific help."""
        self._run_help(
            ["deepagents", "reset", "-h"],
            must_contain="--agent",
            must_not_contain="Start interactive thread",
        )

    def test_threads_list_help(self) -> None:
        """Running `deepagents threads list -h` should show threads list help."""
        self._run_help(
            ["deepagents", "threads", "list", "-h"],
            must_contain="--limit",
            must_not_contain="--sandbox",
        )

    def test_threads_delete_help(self) -> None:
        """Running `deepagents threads delete -h` should show threads delete help."""
        self._run_help(
            ["deepagents", "threads", "delete", "-h"],
            must_contain="delete",
            must_not_contain="--sandbox",
        )


class TestShortFlags:
    """Test that short flag aliases (-a, -M, -v) parse correctly."""

    def test_short_agent_flag(self) -> None:
        """Verify -a sets agent."""
        with patch.object(sys, "argv", ["deepagents", "-a", "mybot"]):
            args = parse_args()
        assert args.agent == "mybot"

    def test_short_model_flag(self) -> None:
        """Verify -M sets model."""
        with patch.object(sys, "argv", ["deepagents", "-M", "gpt-4o"]):
            args = parse_args()
        assert args.model == "gpt-4o"

    def test_agent_default_value(self) -> None:
        """Verify -a defaults to DEFAULT_AGENT_NAME when omitted."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.agent == DEFAULT_AGENT_NAME

    def test_short_version_flag(self) -> None:
        """Verify -v shows version and exits."""
        with (
            patch.object(sys, "argv", ["deepagents", "-v"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code in (0, None)


class TestQuietArg:
    """Tests for -q/--quiet argument parsing."""

    def test_short_flag(self) -> None:
        """Verify -q sets quiet=True."""
        with patch.object(sys, "argv", ["deepagents", "-q", "-n", "task"]):
            args = parse_args()
        assert args.quiet is True

    def test_long_flag(self) -> None:
        """Verify --quiet sets quiet=True."""
        with patch.object(sys, "argv", ["deepagents", "--quiet", "-n", "task"]):
            args = parse_args()
        assert args.quiet is True

    def test_no_flag_defaults_false(self) -> None:
        """Verify quiet is False when not provided."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.quiet is False

    def test_combined_with_non_interactive(self) -> None:
        """Verify -q works alongside -n."""
        with patch.object(sys, "argv", ["deepagents", "-q", "-n", "run tests"]):
            args = parse_args()
        assert args.quiet is True
        assert args.non_interactive_message == "run tests"

    def test_quiet_without_non_interactive_parses(self) -> None:
        """Verify --quiet without -n parses successfully.

        The usage-error guard now lives in `cli_main` (after stdin pipe
        processing), so `parse_args` itself should not reject this combo.
        """
        with patch.object(sys, "argv", ["deepagents", "-q"]):
            args = parse_args()
        assert args.quiet is True
        assert args.non_interactive_message is None


def test_default_agent_name_matches_canonical() -> None:
    """Ensure the duplicated constant in main.py stays in sync with agent.py."""
    assert _DEFAULT_AGENT_NAME == DEFAULT_AGENT_NAME
