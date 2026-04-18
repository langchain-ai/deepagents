"""Tests for agent-friendly CLI improvements.

Covers: --dry-run, idempotency, --stdin, error messages, agents subcommand,
update subcommand, and help screen examples.
"""

import asyncio
import io
import json
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from deepagents_cli.main import parse_args
from deepagents_cli.ui import (
    show_agents_help,
    show_help,
    show_list_help,
    show_reset_help,
    show_skills_delete_help,
    show_skills_info_help,
    show_skills_list_help,
    show_threads_delete_help,
    show_update_help,
)

# ---------------------------------------------------------------------------
# Section 1: Help screen Examples sections
# ---------------------------------------------------------------------------


class TestHelpScreenExamples:
    """Verify Examples sections exist in all subcommand help screens."""

    @staticmethod
    def _render(fn: object) -> str:
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with patch("deepagents_cli.ui.console", test_console):
            fn()  # type: ignore[operator]
        return buf.getvalue()

    def test_list_help_has_examples(self) -> None:
        text = self._render(show_list_help)
        assert "Examples:" in text
        assert "deepagents list" in text
        assert "deepagents list --json" in text

    def test_skills_list_help_has_examples(self) -> None:
        text = self._render(show_skills_list_help)
        assert "Examples:" in text
        assert "deepagents skills list --project" in text
        assert "deepagents skills list --json" in text

    def test_skills_info_help_has_examples(self) -> None:
        text = self._render(show_skills_info_help)
        assert "Examples:" in text
        assert "deepagents skills info web-research" in text
        assert "deepagents skills info my-skill --project" in text

    def test_agents_help_has_examples(self) -> None:
        text = self._render(show_agents_help)
        assert "Examples:" in text
        assert "deepagents agents list" in text
        assert "deepagents agents reset --agent coder" in text

    def test_update_help_has_examples(self) -> None:
        text = self._render(show_update_help)
        assert "Examples:" in text
        assert "deepagents update" in text
        assert "deepagents update --json" in text

    def test_reset_help_has_dry_run(self) -> None:
        text = self._render(show_reset_help)
        assert "--dry-run" in text

    def test_threads_delete_help_has_dry_run(self) -> None:
        text = self._render(show_threads_delete_help)
        assert "--dry-run" in text

    def test_skills_delete_help_has_dry_run(self) -> None:
        text = self._render(show_skills_delete_help)
        assert "--dry-run" in text


# ---------------------------------------------------------------------------
# Section 2: --dry-run for reset
# ---------------------------------------------------------------------------


class TestResetDryRun:
    """Tests for deepagents reset --dry-run."""

    def test_dry_run_text_no_mutation(self, tmp_path: Path) -> None:
        """--dry-run should not remove the agent directory."""
        from deepagents_cli.agent import reset_agent

        agent_dir = tmp_path / "coder"
        agent_dir.mkdir()
        (agent_dir / "AGENTS.md").write_text("original")

        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with (
            patch("deepagents_cli.agent.settings") as mock_settings,
            patch("deepagents_cli.agent.console", test_console),
            patch(
                "deepagents_cli.agent.get_default_coding_instructions",
                return_value="default",
            ),
        ):
            mock_settings.user_deepagents_dir = tmp_path
            reset_agent("coder", dry_run=True)

        assert agent_dir.exists()
        assert (agent_dir / "AGENTS.md").read_text() == "original"
        output = buf.getvalue()
        assert "Would" in output
        assert "No changes made" in output

    def test_dry_run_json(self, tmp_path: Path) -> None:
        """--dry-run --json should include dry_run: true."""
        from deepagents_cli.agent import reset_agent

        agent_dir = tmp_path / "coder"
        agent_dir.mkdir()
        (agent_dir / "AGENTS.md").write_text("original")

        stdout_buf = io.StringIO()
        with (
            patch("deepagents_cli.agent.settings") as mock_settings,
            patch("deepagents_cli.agent.console"),
            patch(
                "deepagents_cli.agent.get_default_coding_instructions",
                return_value="default",
            ),
            patch("sys.stdout", stdout_buf),
        ):
            mock_settings.user_deepagents_dir = tmp_path
            reset_agent("coder", dry_run=True, output_format="json")

        result = json.loads(stdout_buf.getvalue())
        assert result["data"]["dry_run"] is True
        assert agent_dir.exists()


# ---------------------------------------------------------------------------
# Section 2b: --dry-run for threads delete
# ---------------------------------------------------------------------------


class TestThreadsDeleteDryRun:
    """Tests for deepagents threads delete --dry-run."""

    def test_dry_run_text_existing(self) -> None:
        """--dry-run should report what would happen for existing thread."""
        from deepagents_cli import sessions

        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with (
            patch.object(
                sessions,
                "thread_exists",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("deepagents_cli.config.console", test_console),
        ):
            asyncio.run(sessions.delete_thread_command("abc123", dry_run=True))

        output = buf.getvalue()
        assert "Would delete" in output
        assert "No changes made" in output

    def test_dry_run_text_missing(self) -> None:
        """--dry-run should report not found for missing thread."""
        from deepagents_cli import sessions

        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with (
            patch.object(
                sessions,
                "thread_exists",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("deepagents_cli.config.console", test_console),
        ):
            asyncio.run(sessions.delete_thread_command("missing", dry_run=True))

        output = buf.getvalue()
        assert "not found" in output
        assert "No changes made" in output

    def test_dry_run_json(self) -> None:
        """--dry-run --json should include dry_run: true."""
        from deepagents_cli import sessions

        stdout_buf = io.StringIO()
        with (
            patch.object(
                sessions,
                "thread_exists",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("sys.stdout", stdout_buf),
        ):
            asyncio.run(
                sessions.delete_thread_command(
                    "abc123", dry_run=True, output_format="json"
                )
            )

        result = json.loads(stdout_buf.getvalue())
        assert result["data"]["dry_run"] is True


# ---------------------------------------------------------------------------
# Section 3: agents subcommand
# ---------------------------------------------------------------------------


class TestAgentsSubcommand:
    """Tests for the agents resource subcommand."""

    def test_agents_list_parses(self) -> None:
        """Running `deepagents agents list` should parse correctly."""
        with patch.object(sys, "argv", ["deepagents", "agents", "list"]):
            args = parse_args()
        assert args.command == "agents"
        assert args.agents_command == "list"

    def test_agents_ls_alias(self) -> None:
        """Running `deepagents agents ls` should parse as list."""
        with patch.object(sys, "argv", ["deepagents", "agents", "ls"]):
            args = parse_args()
        assert args.command == "agents"
        assert args.agents_command == "ls"

    def test_agents_reset_parses(self) -> None:
        """Running `deepagents agents reset --agent coder` should parse."""
        with patch.object(
            sys, "argv", ["deepagents", "agents", "reset", "--agent", "coder"]
        ):
            args = parse_args()
        assert args.command == "agents"
        assert args.agents_command == "reset"
        assert args.agent == "coder"

    def test_agents_reset_with_target(self) -> None:
        """Running `deepagents agents reset --agent coder --target researcher`."""
        with patch.object(
            sys,
            "argv",
            [
                "deepagents",
                "agents",
                "reset",
                "--agent",
                "coder",
                "--target",
                "researcher",
            ],
        ):
            args = parse_args()
        assert args.source_agent == "researcher"

    def test_agents_reset_dry_run(self) -> None:
        """Running `deepagents agents reset --agent coder --dry-run`."""
        with patch.object(
            sys,
            "argv",
            ["deepagents", "agents", "reset", "--agent", "coder", "--dry-run"],
        ):
            args = parse_args()
        assert args.dry_run is True

    def test_agents_help_exits_clean(self) -> None:
        """Running `deepagents agents -h` should show help and exit 0."""
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=120)
        with (
            patch.object(sys, "argv", ["deepagents", "agents", "-h"]),
            patch("deepagents_cli.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code in (0, None)
        assert "agents" in buf.getvalue().lower()

    def test_top_level_list_rejected(self) -> None:
        """Top-level `deepagents list` should error — use `agents list` instead."""
        with (
            patch.object(sys, "argv", ["deepagents", "list"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Section 4: update subcommand
# ---------------------------------------------------------------------------


class TestUpdateSubcommand:
    """Tests for the update subcommand."""

    def test_update_parses(self) -> None:
        """Running `deepagents update` should parse as command='update'."""
        with patch.object(sys, "argv", ["deepagents", "update"]):
            args = parse_args()
        assert args.command == "update"

    def test_update_help_exits_clean(self) -> None:
        """Running `deepagents update -h` should show help and exit 0."""
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=120)
        with (
            patch.object(sys, "argv", ["deepagents", "update", "-h"]),
            patch("deepagents_cli.ui.console", test_console),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code in (0, None)
        assert "update" in buf.getvalue().lower()


# ---------------------------------------------------------------------------
# Section 5: Idempotency
# ---------------------------------------------------------------------------


class TestSkillsCreateIdempotency:
    """Skills create should be a no-op if skill already exists."""

    def test_already_exists_text_no_error(self, tmp_path: Path) -> None:
        """Re-creating an existing skill should print informational msg, not error."""
        from deepagents_cli.skills.commands import _create

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("existing")

        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        mock_settings = MagicMock()
        mock_settings.project_root = None
        mock_settings.ensure_user_skills_dir.return_value = tmp_path
        with (
            patch("deepagents_cli.config.Settings") as settings_cls,
            patch("deepagents_cli.config.console", test_console),
        ):
            settings_cls.from_environment.return_value = mock_settings
            _create("my-skill", "agent")

        output = buf.getvalue()
        assert "Error" not in output
        assert "already exists" in output

    def test_already_exists_json(self, tmp_path: Path) -> None:
        """Re-creating an existing skill in JSON mode returns already_existed."""
        from deepagents_cli.skills.commands import _create

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("existing")

        stdout_buf = io.StringIO()
        mock_settings = MagicMock()
        mock_settings.project_root = None
        mock_settings.ensure_user_skills_dir.return_value = tmp_path
        with (
            patch("deepagents_cli.config.Settings") as settings_cls,
            patch("deepagents_cli.config.console"),
            patch("sys.stdout", stdout_buf),
        ):
            settings_cls.from_environment.return_value = mock_settings
            _create("my-skill", "agent", output_format="json")

        result = json.loads(stdout_buf.getvalue())
        assert result["data"]["already_existed"] is True


class TestThreadsDeleteIdempotency:
    """Threads delete should be informational (not error) when thread not found."""

    def test_not_found_not_red(self) -> None:
        """Not-found message should not contain Error prefix."""
        from deepagents_cli import sessions

        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with (
            patch.object(
                sessions,
                "delete_thread",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("deepagents_cli.config.console", test_console),
        ):
            asyncio.run(sessions.delete_thread_command("missing"))

        output = buf.getvalue()
        assert "Error" not in output
        assert "not found or already deleted" in output


# ---------------------------------------------------------------------------
# Section 6: --stdin flag
# ---------------------------------------------------------------------------


class TestStdinFlag:
    """Tests for --stdin explicit flag."""

    def test_stdin_flag_parses(self) -> None:
        """Verify --stdin sets stdin=True."""
        with patch.object(sys, "argv", ["deepagents", "--stdin"]):
            args = parse_args()
        assert args.stdin is True

    def test_stdin_default_false(self) -> None:
        """Verify stdin defaults to False."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.stdin is False

    def test_stdin_with_tty_errors(self) -> None:
        """--stdin with a TTY should error."""
        from deepagents_cli.main import apply_stdin_pipe

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True

        args = MagicMock()
        args.stdin = True

        with (
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_cli.config.console"),
            pytest.raises(SystemExit) as exc_info,
        ):
            apply_stdin_pipe(args)

        assert exc_info.value.code == 1

    def test_stdin_omitted_preserves_auto_detect(self) -> None:
        """Omitting --stdin with TTY should return without error."""
        from deepagents_cli.main import apply_stdin_pipe

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True

        args = MagicMock()
        args.stdin = False
        args.non_interactive_message = None
        args.initial_prompt = None

        with patch.object(sys, "stdin", mock_stdin):
            apply_stdin_pipe(args)

        # Should not modify args
        assert args.non_interactive_message is None


# ---------------------------------------------------------------------------
# Section 7: Error messages with corrective hints
# ---------------------------------------------------------------------------


class TestErrorMessageHints:
    """Tests for corrective hints in error messages."""

    def test_no_mcp_conflict_has_hint(self) -> None:
        """--no-mcp + --mcp-config error should include usage examples."""
        from deepagents_cli.main import cli_main

        stderr_buf = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "--no-mcp", "--mcp-config", "/some/path"],
            ),
            patch("deepagents_cli.main.check_cli_dependencies"),
            patch("deepagents_cli.main.apply_stdin_pipe"),
            patch("sys.stderr", stderr_buf),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 2
        output = stderr_buf.getvalue()
        assert "Use one or the other" in output

    def test_quiet_without_n_has_hint(self) -> None:
        """--quiet without -n error should include usage example."""
        from deepagents_cli.main import cli_main

        stderr_buf = io.StringIO()
        with (
            patch.object(sys, "argv", ["deepagents", "--quiet"]),
            patch("deepagents_cli.main.check_cli_dependencies"),
            patch("deepagents_cli.main.apply_stdin_pipe"),
            patch("sys.stderr", stderr_buf),
            pytest.raises(SystemExit),
        ):
            cli_main()

        output = stderr_buf.getvalue()
        assert "deepagents -n" in output

    def test_reset_source_not_found_has_hint(self, tmp_path: Path) -> None:
        """Reset with missing source agent should suggest agents list."""
        from deepagents_cli.agent import reset_agent

        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with (  # separate to satisfy PT012
            patch("deepagents_cli.agent.settings") as mock_settings,
            patch("deepagents_cli.agent.console", test_console),
        ):
            mock_settings.user_deepagents_dir = tmp_path
            with pytest.raises(SystemExit):
                reset_agent("coder", "nonexistent")

        output = buf.getvalue()
        assert "deepagents agents list" in output


# ---------------------------------------------------------------------------
# Drift detection: new flags in argparse should appear in help screens
# ---------------------------------------------------------------------------


class TestHelpScreenDriftExtended:
    """Extended drift detection for new subcommands."""

    def test_show_help_includes_agents_subcommand(self) -> None:
        """show_help should mention the agents subcommand."""
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with patch("deepagents_cli.ui.console", test_console):
            show_help()
        assert "agents" in buf.getvalue()

    def test_show_help_includes_update_subcommand(self) -> None:
        """show_help should mention the update subcommand."""
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with patch("deepagents_cli.ui.console", test_console):
            show_help()
        assert "update" in buf.getvalue()

    def test_show_help_includes_stdin(self) -> None:
        """show_help should mention --stdin."""
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with patch("deepagents_cli.ui.console", test_console):
            show_help()
        assert "--stdin" in buf.getvalue()

    def test_all_parser_flags_appear_in_help(self) -> None:
        """Every top-level --flag in argparse must appear in show_help()."""
        stderr_buf = io.StringIO()
        with (
            patch.object(sys, "argv", ["deepagents", "--_x_"]),
            patch("sys.stderr", stderr_buf),
            pytest.raises(SystemExit),
        ):
            parse_args()
        usage_text = stderr_buf.getvalue()

        help_buf = io.StringIO()
        test_console = Console(file=help_buf, highlight=False, width=200)
        with patch("deepagents_cli.ui.console", test_console):
            show_help()
        help_text = help_buf.getvalue()

        parser_flags = set(re.findall(r"--[\w][\w-]*", usage_text))
        help_flags = set(re.findall(r"--[\w][\w-]*", help_text))
        parser_flags.discard("--_x_")

        missing = parser_flags - help_flags
        assert not missing, (
            f"Flags in argparse but missing from show_help(): {missing}\n"
            "Add them to the Options section in ui.show_help()."
        )
