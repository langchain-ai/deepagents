"""Tests for CLI argument parsing."""

import io
import re
import sys
from unittest.mock import AsyncMock, patch

import pytest
from rich.console import Console

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.main import parse_args
from deepagents_code.ui import show_help, show_threads_list_help


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


def test_plugin_command_parses_when_experimental_features_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EXPERIMENTAL, raising=False)
    with patch.object(sys, "argv", ["deepagents", "plugin", "list"]):
        args = parse_args()
    assert args.command == "plugin"
    assert args.plugin_command == "list"


class TestInitialSkillArg:
    """Tests for `--skill` startup skill argument."""

    def test_flag_sets_initial_skill(self) -> None:
        """Verify `--skill` stores the requested skill name."""
        with patch.object(sys, "argv", ["deepagents", "--skill", "code-review"]):
            args = parse_args()
        assert args.initial_skill == "code-review"

    def test_with_message(self) -> None:
        """Verify `--skill` works alongside `-m`."""
        with patch.object(
            sys,
            "argv",
            ["deepagents", "--skill", "code-review", "-m", "review this patch"],
        ):
            args = parse_args()
        assert args.initial_skill == "code-review"
        assert args.initial_prompt == "review this patch"


class TestMaxRetriesArg:
    """Tests for `--max-retries` argument."""

    def test_valid_int_passes_through(self) -> None:
        """`--max-retries` stores a non-negative integer."""
        with patch.object(sys, "argv", ["deepagents", "--max-retries", "3"]):
            args = parse_args()
        assert args.max_retries == 3

    def test_zero_passes_through(self) -> None:
        """`--max-retries 0` is valid."""
        with patch.object(sys, "argv", ["deepagents", "--max-retries", "0"]):
            args = parse_args()
        assert args.max_retries == 0

    def test_negative_rejected(self) -> None:
        """Negative retry counts are rejected by argparse."""
        with (
            patch.object(sys, "argv", ["deepagents", "--max-retries", "-1"]),
            pytest.raises(SystemExit),
        ):
            parse_args()

    def test_non_int_rejected(self) -> None:
        """Non-integer retry counts are rejected by argparse."""
        with (
            patch.object(sys, "argv", ["deepagents", "--max-retries=foo"]),
            pytest.raises(SystemExit),
        ):
            parse_args()


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

    def test_flag_sets_startup_cmd(self) -> None:
        """Verify `--startup-cmd` stores the requested command."""
        with patch.object(sys, "argv", ["deepagents", "--startup-cmd", "git status"]):
            args = parse_args()
        assert args.startup_cmd == "git status"

    def test_no_flag(self) -> None:
        """Verify `startup_cmd` defaults to `None`."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.startup_cmd is None

    def test_with_non_interactive(self) -> None:
        """Verify `--startup-cmd` works alongside `-n`."""
        with patch.object(
            sys,
            "argv",
            [
                "deepagents",
                "--startup-cmd",
                "echo hi",
                "-n",
                "do the thing",
            ],
        ):
            args = parse_args()
        assert args.startup_cmd == "echo hi"
        assert args.non_interactive_message == "do the thing"


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

    def test_short_agent_flag(self) -> None:
        """Verify -a sets agent."""
        with patch.object(sys, "argv", ["deepagents", "-a", "mybot"]):
            args = parse_args()
        assert args.agent == "mybot"

    def test_short_model_flag(self) -> None:
        """Verify -M sets model."""
        with patch.object(sys, "argv", ["deepagents", "-M", "gpt-5.5"]):
            args = parse_args()
        assert args.model == "gpt-5.5"

    def test_agent_default_value(self) -> None:
        """Verify -a is `None` when omitted so downstream fallback can run.

        The `[agents].recent` / default-agent fallback lives in
        `_resolve_agent_arg`, not argparse — argparse must leave the slot
        empty so the resolver can distinguish "user didn't pass -a" from
        "user explicitly passed the default name".
        """
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.agent is None

    def test_short_version_flag(self) -> None:
        """Verify -v shows version and exits."""
        with (
            patch.object(sys, "argv", ["deepagents", "-v"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code in (0, None)

    def test_short_auto_approve_flag(self) -> None:
        """Verify -y sets auto_approve."""
        with patch.object(sys, "argv", ["deepagents", "-y"]):
            args = parse_args()
        assert args.auto_approve is True

    def test_short_shell_allow_list_flag(self) -> None:
        """Verify -S sets shell_allow_list."""
        with patch.object(sys, "argv", ["deepagents", "-S", "ls,cat"]):
            args = parse_args()
        assert args.shell_allow_list == "ls,cat"


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


class TestNoMcpArg:
    """Tests for --no-mcp argument parsing."""

    def test_no_mcp_flag_parsed(self) -> None:
        """Verify --no-mcp sets no_mcp=True."""
        with patch.object(sys, "argv", ["deepagents", "--no-mcp"]):
            args = parse_args()
        assert args.no_mcp is True

    def test_no_mcp_default_false(self) -> None:
        """Verify no_mcp defaults to False."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.no_mcp is False

    def test_no_mcp_and_mcp_config_mutual_exclusion(self) -> None:
        """--no-mcp + --mcp-config should exit with code 2."""
        from deepagents_code.main import cli_main

        with (  # noqa: SIM117  # separate to satisfy PT012
            patch.object(
                sys,
                "argv",
                ["deepagents", "--no-mcp", "--mcp-config", "/some/path"],
            ),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.main.apply_stdin_pipe"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
        assert exc_info.value.code == 2


class TestConfigCommandDispatch:
    """Tests for `cli_main()` dispatch of `dcode config` subcommands."""

    def test_config_command_exits_before_stdin_pipe(self) -> None:
        """`dcode config` is headless and must not read stdin."""
        from deepagents_code.main import cli_main

        with (
            patch.object(
                sys,
                "argv",
                [
                    "deepagents",
                    "config",
                    "get",
                    "interpreter.memory_limit_mb",
                    "--json",
                ],
            ),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch(
                "deepagents_code.main.apply_stdin_pipe",
                side_effect=AssertionError("config command read stdin"),
            ) as stdin_mock,
            patch(
                "deepagents_code.client.commands.config.run_config_command",
                return_value=0,
            ) as config_mock,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        stdin_mock.assert_not_called()
        config_mock.assert_called_once()
        args = config_mock.call_args.args[0]
        assert args.command == "config"
        assert args.config_command == "get"


class TestMcpCommandDispatch:
    """Tests for `cli_main()` dispatch of `dcode mcp` subcommands."""

    def test_mcp_login_uses_top_level_mcp_config_as_fallback(self) -> None:
        """`dcode --mcp-config PATH mcp login NAME` propagates PATH to login."""
        from deepagents_code.main import cli_main

        with (
            patch.object(
                sys,
                "argv",
                [
                    "deepagents",
                    "--mcp-config",
                    "/global/config.json",
                    "mcp",
                    "login",
                    "notion",
                ],
            ),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.main.apply_stdin_pipe"),
            patch(
                "deepagents_code.client.commands.mcp.run_mcp_login",
                new=AsyncMock(return_value=0),
            ) as mock_login,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        mock_login.assert_awaited_once_with(
            server="notion",
            config_path="/global/config.json",
        )

    def test_mcp_login_subcommand_mcp_config_wins_over_top_level(self) -> None:
        """Subcommand `--mcp-config` overrides the top-level value."""
        from deepagents_code.main import cli_main

        with (
            patch.object(
                sys,
                "argv",
                [
                    "deepagents",
                    "--mcp-config",
                    "/global/config.json",
                    "mcp",
                    "login",
                    "notion",
                    "--mcp-config",
                    "/subcommand/config.json",
                ],
            ),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.main.apply_stdin_pipe"),
            patch(
                "deepagents_code.client.commands.mcp.run_mcp_login",
                new=AsyncMock(return_value=0),
            ) as mock_login,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        mock_login.assert_awaited_once_with(
            server="notion",
            config_path="/subcommand/config.json",
        )

    def test_mcp_config_subcommand_prints_discovery_paths(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`dcode mcp config` prints each discovery path."""
        from deepagents_code.main import cli_main

        with (
            patch.object(sys, "argv", ["deepagents", "mcp", "config"]),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.main.apply_stdin_pipe"),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "~/.deepagents/.mcp.json" in out
        assert "<project-root>/.deepagents/.mcp.json" in out
        assert "<project-root>/.mcp.json" in out

    def test_mcp_login_subcommand_mcp_config_only(self) -> None:
        """`dcode mcp login NAME --mcp-config PATH` passes PATH through.

        Covers the subcommand-only path (no top-level value) so a future
        reorder of the `or`-precedence in dispatch fails loudly.
        """
        from deepagents_code.main import cli_main

        with (
            patch.object(
                sys,
                "argv",
                [
                    "deepagents",
                    "mcp",
                    "login",
                    "notion",
                    "--mcp-config",
                    "/sub/config.json",
                ],
            ),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.main.apply_stdin_pipe"),
            patch(
                "deepagents_code.client.commands.mcp.run_mcp_login",
                new=AsyncMock(return_value=0),
            ) as mock_login,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        mock_login.assert_awaited_once_with(
            server="notion",
            config_path="/sub/config.json",
        )

    def test_mcp_login_rejects_old_config_flag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The renamed `--config` flag is rejected by argparse.

        Documents the intentional backcompat break (renamed to
        `--mcp-config`) and prevents a stealth re-introduction of the
        alias.
        """
        from deepagents_code.main import cli_main

        with (
            patch.object(
                sys,
                "argv",
                [
                    "deepagents",
                    "mcp",
                    "login",
                    "notion",
                    "--config",
                    "/some/path.json",
                ],
            ),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.main.apply_stdin_pipe"),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "unrecognized arguments" in err
        assert "--config" in err

    def test_mcp_config_marker_reflects_filesystem(
        self,
        tmp_path: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`run_mcp_config()` marks files [found] / [missing] accurately.

        Points `Path.home()` and `find_project_root()` at an isolated
        tmp_path, creates the user-level file only, then asserts the
        marker on each row.
        """
        import pathlib

        from deepagents_code.client.commands.mcp import run_mcp_config

        fake_home = pathlib.Path(str(tmp_path)) / "home"
        fake_project = pathlib.Path(str(tmp_path)) / "project"
        (fake_home / ".deepagents").mkdir(parents=True)
        (fake_home / ".deepagents" / ".mcp.json").write_text("{}")
        fake_project.mkdir()

        monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)
        monkeypatch.chdir(fake_project)
        monkeypatch.setattr(
            "deepagents_code.project_utils.find_project_root",
            lambda: fake_project,
        )

        exit_code = run_mcp_config()

        assert exit_code == 0
        out = capsys.readouterr().out
        user_line = next(
            line for line in out.splitlines() if "~/.deepagents/.mcp.json" in line
        )
        project_root_line = next(
            line
            for line in out.splitlines()
            if "<project-root>/.mcp.json" in line
            and "<project-root>/.deepagents" not in line
        )
        project_subdir_line = next(
            line
            for line in out.splitlines()
            if "<project-root>/.deepagents/.mcp.json" in line
        )
        assert "found" in user_line
        assert "missing" in project_root_line
        assert "missing" in project_subdir_line


class TestAutoUpdateArg:
    """Tests for --auto-update argument parsing."""

    def test_flag_parsed(self) -> None:
        """Verify --auto-update sets auto_update=True."""
        with patch.object(sys, "argv", ["deepagents", "--auto-update"]):
            args = parse_args()
        assert args.auto_update is True

    def test_default_false(self) -> None:
        """Verify auto_update defaults to False."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.auto_update is False


class TestHelpScreenDrift:
    """Ensure show_help() stays in sync with argparse flag definitions.

    The help screen in `ui.show_help()` is hand-maintained separately from
    the argparse definitions in `main.parse_args()`.  This test catches
    drift — e.g. a new flag added to argparse but forgotten in the help screen.
    """

    def test_all_parser_flags_appear_in_help(self) -> None:
        """Every top-level --flag in argparse must appear in show_help()."""
        # 1. Trigger argparse usage line by passing an unrecognized flag.
        #    argparse prints the full usage (all flags) to stderr, then exits.
        stderr_buf = io.StringIO()
        with (
            patch.object(sys, "argv", ["deepagents", "--_x_"]),
            patch("sys.stderr", stderr_buf),
            pytest.raises(SystemExit),
        ):
            parse_args()
        usage_text = stderr_buf.getvalue()

        # 2. Render show_help() to a string.
        help_buf = io.StringIO()
        test_console = Console(file=help_buf, highlight=False, width=200)
        with patch("deepagents_code.ui.console", test_console):
            show_help()
        help_text = help_buf.getvalue()

        # 3. Extract --long-form flags from both and compare.
        parser_flags = set(re.findall(r"--[\w][\w-]*", usage_text))
        help_flags = set(re.findall(r"--[\w][\w-]*", help_text))

        parser_flags.discard("--_x_")  # remove the fake trigger flag

        missing = parser_flags - help_flags
        assert not missing, (
            f"Flags in argparse but missing from show_help(): {missing}\n"
            "Add them to the Options section in ui.show_help()."
        )

    def test_threads_list_flags_appear_in_help(self) -> None:
        """Every `threads list`-specific --flag must appear in show_threads_list_help().

        We capture the argparse -h output for the subcommand, then compare
        only the optional-arguments section (after "options:") to avoid
        matching inherited global flags in the usage line.
        """
        stdout_buf = io.StringIO()
        with (
            patch.object(sys, "argv", ["deepagents", "threads", "list", "-h"]),
            patch("sys.stdout", stdout_buf),
            patch("deepagents_code.ui.console", Console(file=io.StringIO())),
            pytest.raises(SystemExit),
        ):
            parse_args()
        raw = stdout_buf.getvalue()

        # Only look at the "options:" section to avoid inherited global flags
        options_section = raw.split("options:")[-1] if "options:" in raw else raw
        parser_flags = set(re.findall(r"--[\w][\w-]*", options_section))
        parser_flags.discard("--help")

        help_buf = io.StringIO()
        test_console = Console(file=help_buf, highlight=False, width=200)
        with patch("deepagents_code.ui.console", test_console):
            show_threads_list_help()
        help_flags = set(re.findall(r"--[\w][\w-]*", help_buf.getvalue()))

        missing = parser_flags - help_flags
        assert not missing, (
            f"Flags in argparse but missing from show_threads_list_help(): {missing}\n"
            "Add them to the Options section in ui.show_threads_list_help()."
        )


class TestJsonArg:
    """Tests for `--json` argument parsing."""

    def test_default_text(self) -> None:
        """Verify output_format defaults to text."""
        with patch.object(sys, "argv", ["deepagents"]):
            args = parse_args()
        assert args.output_format == "text"

    def test_json_shortcut(self) -> None:
        """Verify --json sets output_format to json."""
        with patch.object(sys, "argv", ["deepagents", "--json"]):
            args = parse_args()
        assert args.output_format == "json"

    def test_json_before_subcommand(self) -> None:
        """Verify --json works before a subcommand."""
        with patch.object(sys, "argv", ["deepagents", "--json", "agents", "list"]):
            args = parse_args()
        assert args.command == "agents"
        assert args.output_format == "json"

    def test_json_after_subcommand(self) -> None:
        """Verify --json works after a subcommand."""
        with patch.object(sys, "argv", ["deepagents", "agents", "list", "--json"]):
            args = parse_args()
        assert args.command == "agents"
        assert args.output_format == "json"

    def test_output_format_flag_removed(self) -> None:
        """Verify --output-format is no longer accepted."""
        with (
            patch.object(sys, "argv", ["deepagents", "--output-format", "json"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2

    def test_json_after_nested_subcommand(self) -> None:
        """Verify --json works after nested subcommands."""
        with patch.object(sys, "argv", ["deepagents", "skills", "list", "--json"]):
            args = parse_args()
        assert args.command == "skills"
        assert args.skills_command == "list"
        assert args.output_format == "json"
