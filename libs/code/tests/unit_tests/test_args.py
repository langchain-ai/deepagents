"""Tests for CLI argument parsing."""

import argparse
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from deepagents_code._constants import DEFAULT_AGENT_NAME
from deepagents_code.main import _resolve_agent_arg, parse_args


class TestInitialPromptArg:
    """Tests for -m/--message initial prompt argument."""


class TestInitialSkillArg:
    """Tests for `--skill` startup skill argument."""


class TestSummarizationModelArg:
    def test_flag_sets_model_independently(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "deepagents",
                "--model",
                "anthropic:claude-sonnet-4-5",
                "--summarization-model",
                "openai:gpt-5.4-mini",
            ],
        ):
            args = parse_args()
        assert args.model == "anthropic:claude-sonnet-4-5"
        assert args.summarization_model == "openai:gpt-5.4-mini"

    def test_flag_defaults_to_none(self) -> None:
        with patch.object(sys, "argv", ["deepagents"]):
            assert parse_args().summarization_model is None


class TestMaxRetriesArg:
    """Tests for `--max-retries` argument."""


class TestSandboxSnapshotNameArg:
    """Tests for `--sandbox-snapshot-name` argument."""

    def test_flag_sets_snapshot_name(self) -> None:
        """Verify `--sandbox-snapshot-name` stores the requested snapshot name."""
        with patch.object(
            sys,
            "argv",
            [
                "deepagents",
                "--sandbox",
                "langsmith",
                "--sandbox-snapshot-name",
                "custom-snap",
            ],
        ):
            args = parse_args()
        assert args.sandbox_snapshot_name == "custom-snap"

    def test_no_flag(self) -> None:
        """Verify `sandbox_snapshot_name` defaults to `None`."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.sandbox_snapshot_name is None

    def test_snapshot_name_without_langsmith_or_runloop_errors(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--sandbox-snapshot-name` without a supporting sandbox errors out."""
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "--sandbox-snapshot-name", "custom-snap"],
            ),
            pytest.raises(SystemExit),
        ):
            parse_args()
        assert "requires a --sandbox provider" in capsys.readouterr().err

    def test_snapshot_name_with_runloop(self) -> None:
        """`--sandbox-snapshot-name` is allowed with `--sandbox runloop`."""
        with patch.object(
            sys,
            "argv",
            [
                "deepagents",
                "--sandbox",
                "runloop",
                "--sandbox-snapshot-name",
                "custom-bp",
            ],
        ):
            args = parse_args()
        assert args.sandbox == "runloop"
        assert args.sandbox_snapshot_name == "custom-bp"


class TestSandboxArg:
    """Tests for `--sandbox` provider choices."""

    def test_vercel_choice(self) -> None:
        """Verify `--sandbox vercel` parses."""
        with patch.object(sys, "argv", ["deepagents", "--sandbox", "vercel"]):
            args = parse_args()
        assert args.sandbox == "vercel"


class TestStartupCmdArg:
    """Tests for `--startup-cmd` pre-prompt shell command argument."""


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
            patch("deepagents_code.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()

        assert exc_info.value.code in (0, None)
        output = buf.getvalue()

        # Should contain global help content
        assert "deepagents" in output.lower()
        assert "--help" in output


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
            patch("deepagents_code.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()

        assert exc_info.value.code in (0, None)
        output = buf.getvalue()
        assert must_contain in output
        assert must_not_contain not in output

    def test_agents_list_help(self) -> None:
        """Running `deepagents agents list -h` should show list-specific help."""
        self._run_help(
            ["deepagents", "agents", "list", "-h"],
            must_contain="List all agents",
            must_not_contain="--sandbox",
        )

    def test_agents_reset_help(self) -> None:
        """Running `deepagents agents reset -h` should show reset-specific help."""
        self._run_help(
            ["deepagents", "agents", "reset", "-h"],
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
    """Test that short flag aliases (-a, -M, -S, -v, -y) parse correctly."""

    def test_short_shell_allow_list_flag(self) -> None:
        """Verify -S sets shell_allow_list."""
        with patch.object(sys, "argv", ["deepagents", "-S", "ls,cat"]):
            args = parse_args()
        assert args.shell_allow_list == "ls,cat"


class TestQuietArg:
    """Tests for -q/--quiet argument parsing."""


class TestNoMcpArg:
    """Tests for --no-mcp argument parsing."""


class TestConfigCommandDispatch:
    """Tests for `cli_main()` dispatch of `dcode config` subcommands."""


class TestMcpCommandDispatch:
    """Tests for `cli_main()` dispatch of `dcode mcp` subcommands."""


class TestAutoUpdateArg:
    """Tests for --auto-update argument parsing."""


class TestRecursionLimitArg:
    """Tests for the --recursion-limit override flag."""


class TestJsonArg:
    """Tests for `--json` argument parsing."""


class TestReservedAgentArg:
    """`-a plugins` must fail at the CLI, not inside agent construction.

    `agent.create_cli_agent` calls `settings.ensure_agent_dir(assistant_id)`
    with no handler, so a reserved name surfaced to the user as an unhandled
    `ValueError` from server startup.
    """

    @pytest.mark.parametrize("name", ["bin", "plugins", "conversation_history"])
    def test_reserved_name_exits_with_a_message(
        self, name: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(agent=name, resume_thread=None)

        with pytest.raises(SystemExit) as exc_info:
            _resolve_agent_arg(args)

        assert exc_info.value.code == 2
        output = capsys.readouterr().out
        assert name in output
        assert "reserved" in output

    def test_ordinary_name_passes_through(self) -> None:
        args = argparse.Namespace(agent="coder", resume_thread=None)

        assert _resolve_agent_arg(args) == "coder"


class TestStaleStoredAgent:
    """A stored `[agents].recent` must never break every launch.

    `bin/` and `plugins/` are real directories under the profile root, so the
    `is_dir()` staleness check accepted them and the launch then failed in
    `get_agent_dir`.
    """

    def test_a_reserved_stored_name_falls_back_to_the_default(
        self, tmp_path: Path
    ) -> None:
        """End to end through the resolver, with the directory actually present."""
        (tmp_path / "plugins").mkdir()
        args = argparse.Namespace(agent=None, resume_thread=None)

        with (
            patch("deepagents_code.main.get_deepagents_home", return_value=tmp_path),
            patch("deepagents_code.model_config.load_default_agent", return_value=None),
            patch(
                "deepagents_code.model_config.load_recent_agent",
                return_value="plugins",
            ),
        ):
            assert _resolve_agent_arg(args) == DEFAULT_AGENT_NAME
