"""Tests for command-line argument parsing."""

import argparse
import asyncio
import io
import os
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.config import parse_shell_allow_list
from deepagents_code.main import apply_stdin_pipe, parse_args

MockArgvType = Callable[..., AbstractContextManager[object]]


@pytest.fixture
def mock_argv() -> MockArgvType:
    """Factory fixture to mock sys.argv with given arguments."""

    def _mock_argv(*args: str) -> AbstractContextManager[object]:
        return patch.object(sys, "argv", ["deepagents", *args])

    return _mock_argv


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--shell-allow-list", "ls,cat,grep"], "ls,cat,grep"),
        (["--shell-allow-list", "ls, cat , grep"], "ls, cat , grep"),
        (["--shell-allow-list", "ls"], "ls"),
        (
            ["--shell-allow-list", "ls,cat,grep,pwd,echo,head,tail,find,wc,tree"],
            "ls,cat,grep,pwd,echo,head,tail,find,wc,tree",
        ),
    ],
)
def test_shell_allow_list_argument(
    args: list[str], expected: str, mock_argv: MockArgvType
) -> None:
    """Test --shell-allow-list argument with various values."""
    with mock_argv(*args):
        parsed_args = parse_args()
        assert hasattr(parsed_args, "shell_allow_list")
        assert parsed_args.shell_allow_list == expected


def test_shell_allow_list_not_specified(mock_argv: MockArgvType) -> None:
    """Test that shell_allow_list is None when not specified."""
    with mock_argv():
        parsed_args = parse_args()
        assert hasattr(parsed_args, "shell_allow_list")
        assert parsed_args.shell_allow_list is None


def test_parse_args_does_not_prepend_managed_bin(
    monkeypatch: pytest.MonkeyPatch, mock_argv: MockArgvType
) -> None:
    """PATH is only changed after the managed ripgrep binary is validated."""
    path = f"/usr/bin{os.pathsep}/bin"
    monkeypatch.setenv("PATH", path)

    with mock_argv():
        parse_args()

    assert os.environ["PATH"] == path


def test_headless_installs_ripgrep_when_warning_is_suppressed() -> None:
    """Suppressed warning state must not skip headless managed `rg` install."""
    from deepagents_code.main import cli_main

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    ensure = AsyncMock(return_value=Path("/managed/rg"))
    prepend = MagicMock()
    with (
        patch.object(sys, "argv", ["deepagents", "-n", "task"]),
        patch.object(sys, "stdin", mock_stdin),
        patch("deepagents_code.main.check_optional_tools", return_value=[]),
        patch(
            "deepagents_code.main._should_ensure_managed_ripgrep",
            return_value=True,
        ),
        patch("deepagents_code.managed_tools.ensure_ripgrep", ensure),
        patch(
            "deepagents_code.managed_tools.managed_rg_path",
            return_value=Path("/managed/rg"),
        ),
        patch("deepagents_code.managed_tools.prepend_managed_bin_to_path", prepend),
        patch(
            "deepagents_code.client.non_interactive.run_non_interactive",
            new_callable=AsyncMock,
            return_value=0,
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    ensure.assert_awaited_once()
    prepend.assert_called_once()


def test_shell_allow_list_combined_with_other_args(mock_argv: MockArgvType) -> None:
    """Test that shell-allow-list works with other arguments."""
    with mock_argv(
        "--shell-allow-list", "ls,cat", "--model", "gpt-5.5", "--auto-approve"
    ):
        parsed_args = parse_args()
        assert parsed_args.shell_allow_list == "ls,cat"
        assert parsed_args.model == "gpt-5.5"
        assert parsed_args.auto_approve is True


class TestAutoApproveArgument:
    """Tests for -y / --auto-approve parsing and its config.toml default."""

    def test_flag_sets_true(self, mock_argv: MockArgvType) -> None:
        """Passing -y sets auto_approve to True."""
        with mock_argv("-y"):
            assert parse_args().auto_approve is True

    def test_long_flag_sets_true(self, mock_argv: MockArgvType) -> None:
        """Passing --auto-approve sets auto_approve to True."""
        with mock_argv("--auto-approve"):
            assert parse_args().auto_approve is True

    def test_omitted_is_none(self, mock_argv: MockArgvType) -> None:
        """Omitting the flag leaves auto_approve as None so config.toml decides."""
        with mock_argv():
            assert parse_args().auto_approve is None


class TestResolveAutoApprove:
    """Tests for `_resolve_auto_approve` (flag vs. `[startup].mode` precedence)."""

    def test_explicit_flag_wins_without_consulting_config(self) -> None:
        """An explicit -y (True) resolves True and never reads config."""
        from deepagents_code.main import _resolve_auto_approve

        args = argparse.Namespace(auto_approve=True)
        with patch("deepagents_code.model_config.load_startup_mode") as mock_load:
            assert _resolve_auto_approve(args) is True
        mock_load.assert_not_called()

    def test_omitted_flag_manual_config_resolves_false(self) -> None:
        """No flag + `[startup].mode = manual` keeps approvals on (False)."""
        from deepagents_code.main import _resolve_auto_approve

        args = argparse.Namespace(auto_approve=None)
        with patch(
            "deepagents_code.model_config.load_startup_mode",
            return_value="manual",
        ):
            assert _resolve_auto_approve(args) is False

    def test_omitted_flag_dangerously_auto_config_resolves_true(self) -> None:
        """No flag + `[startup].mode = dangerously-auto` auto-approves (True)."""
        from deepagents_code.main import _resolve_auto_approve

        args = argparse.Namespace(auto_approve=None)
        with patch(
            "deepagents_code.model_config.load_startup_mode",
            return_value="dangerously-auto",
        ):
            assert _resolve_auto_approve(args) is True


class TestAutoApproveHeadlessValidation:
    """Tests that headless mode rejects the interactive approval flag."""

    def test_rejects_explicit_non_interactive_mode(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`-n` must reject `--auto-approve` instead of silently ignoring it."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "--auto-approve", "-n", "do the thing"],
            ),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 2
        stderr = capsys.readouterr().err
        assert "--auto-approve is only supported in interactive mode" in stderr
        assert "--shell-allow-list" in stderr

    def test_rejects_piped_stdin_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Piped stdin must reject the flag after selecting headless mode."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "do the thing"
        with (
            patch.object(sys, "argv", ["deepagents", "--auto-approve"]),
            patch.object(sys, "stdin", mock_stdin),
            patch("os.open", side_effect=OSError("No controlling terminal")),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 2
        stderr = capsys.readouterr().err
        assert "--auto-approve is only supported in interactive mode" in stderr
        assert "--shell-allow-list" in stderr

    def test_accepts_auto_approve_in_interactive_mode(self) -> None:
        """`--auto-approve` must still be honored on an interactive launch.

        The guard rejects only when `args.non_interactive_message` is also set.
        Without that conjunct it would wrongly reject `dcode -m ... -y`; this
        pins the interactive path so a dropped conjunct fails loudly instead of
        silently breaking the flag's primary use. Also asserts the resolved
        value flows through to the TUI (`auto_approve=True`).
        """
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True

        fake_result = MagicMock()
        fake_result.return_code = 0
        fake_result.thread_id = None
        fake_result.update_available = (False, None)
        fake_result.session_stats = MagicMock(request_count=0)
        run_tui = AsyncMock(return_value=fake_result)

        with (
            patch.object(sys, "argv", ["deepagents", "--auto-approve", "-m", "hello"]),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.run_textual_cli_async", run_tui),
            patch("deepagents_code.main._run_startup_auto_update"),
            patch("deepagents_code.main._resolve_agent_arg", return_value="agent"),
            patch("deepagents_code.main._check_mcp_project_trust", return_value=False),
            patch(
                "deepagents_code.main._resolve_interpreter_enabled",
                return_value=False,
            ),
            patch("deepagents_code.main._print_session_stats"),
            patch(
                "deepagents_code.main._should_check_teardown_thread",
                return_value=False,
            ),
        ):
            cli_main()

        run_tui.assert_awaited_once()
        await_args = run_tui.await_args
        assert await_args is not None
        assert await_args.kwargs["auto_approve"] is True


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("ls,cat,grep", ["ls", "cat", "grep"]),
        ("ls , cat , grep", ["ls", "cat", "grep"]),
        ("ls,cat,grep,", ["ls", "cat", "grep"]),
        ("ls", ["ls"]),
    ],
)
def test_shell_allow_list_string_parsing(input_str: str, expected: list[str]) -> None:
    """Test parsing shell-allow-list string into list using actual config function."""
    result = parse_shell_allow_list(input_str)
    assert result == expected


class TestNonInteractiveArgument:
    """Tests for -n / --non-interactive argument parsing."""

    def test_short_flag(self, mock_argv: MockArgvType) -> None:
        """Test -n flag stores the message."""
        with mock_argv("-n", "run tests"):
            parsed = parse_args()
            assert parsed.non_interactive_message == "run tests"

    def test_long_flag(self, mock_argv: MockArgvType) -> None:
        """Test --non-interactive flag stores the message."""
        with mock_argv("--non-interactive", "fix the bug"):
            parsed = parse_args()
            assert parsed.non_interactive_message == "fix the bug"

    def test_not_specified_is_none(self, mock_argv: MockArgvType) -> None:
        """Test non_interactive_message is None when not provided."""
        with mock_argv():
            parsed = parse_args()
            assert parsed.non_interactive_message is None

    def test_combined_with_shell_allow_list(self, mock_argv: MockArgvType) -> None:
        """Test -n works alongside --shell-allow-list."""
        with mock_argv("-n", "deploy app", "--shell-allow-list", "ls,cat"):
            parsed = parse_args()
            assert parsed.non_interactive_message == "deploy app"
            assert parsed.shell_allow_list == "ls,cat"

    def test_combined_with_sandbox_setup(self, mock_argv: MockArgvType) -> None:
        """Test -n works alongside --sandbox and --sandbox-setup."""
        with mock_argv(
            "-n",
            "run task",
            "--sandbox",
            "modal",
            "--sandbox-setup",
            "/path/to/setup.sh",
        ):
            parsed = parse_args()
            assert parsed.non_interactive_message == "run task"
            assert parsed.sandbox == "modal"
            assert parsed.sandbox_setup == "/path/to/setup.sh"


class TestSandboxArgument:
    """Tests for `--sandbox` resolution and registry validation."""

    def test_builtin_provider_accepted(self, mock_argv: MockArgvType) -> None:
        with mock_argv("-n", "task", "--sandbox", "daytona"):
            parsed = parse_args()
            assert parsed.sandbox == "daytona"

    def test_default_when_omitted_is_none_string(self, mock_argv: MockArgvType) -> None:
        with mock_argv("-n", "task"):
            parsed = parse_args()
            assert parsed.sandbox == "none"

    def test_unknown_provider_errors(self, mock_argv: MockArgvType) -> None:
        with (
            mock_argv("-n", "task", "--sandbox", "acme"),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2

    def test_unknown_provider_error_includes_guidance(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The unknown-provider error explains how to install or configure."""
        with (
            mock_argv("-n", "task", "--sandbox", "acme"),
            pytest.raises(SystemExit),
        ):
            parse_args()
        err = capsys.readouterr().err
        assert "/install <package-name> --package" in err
        assert "[sandboxes.providers.acme]" in err
        # The error must not fabricate a specific package name.
        assert "acme-dcode-sandbox" not in err

    def test_config_provider_accepted(
        self, mock_argv: MockArgvType, tmp_path: Path
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            '[sandboxes.providers.acme]\nclass_path = "acme:Provider"\n',
            encoding="utf-8",
        )
        with (
            patch(
                "deepagents_code.integrations.sandbox_config.DEFAULT_CONFIG_PATH",
                config,
            ),
            mock_argv("-n", "task", "--sandbox", "acme"),
        ):
            parsed = parse_args()
            assert parsed.sandbox == "acme"

    def test_bare_sandbox_resolves_config_default(
        self, mock_argv: MockArgvType, tmp_path: Path
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            '[sandboxes]\ndefault = "acme"\n\n'
            '[sandboxes.providers.acme]\nclass_path = "acme:Provider"\n',
            encoding="utf-8",
        )
        with (
            patch(
                "deepagents_code.integrations.sandbox_config.DEFAULT_CONFIG_PATH",
                config,
            ),
            mock_argv("-n", "task", "--sandbox"),
        ):
            parsed = parse_args()
            assert parsed.sandbox == "acme"

    def test_bare_sandbox_without_default_errors(
        self, mock_argv: MockArgvType, tmp_path: Path
    ) -> None:
        config = tmp_path / "config.toml"
        with (
            patch(
                "deepagents_code.integrations.sandbox_config.DEFAULT_CONFIG_PATH",
                config,
            ),
            mock_argv("-n", "task", "--sandbox"),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2

    def test_snapshot_name_rejected_for_unsupported_provider(
        self, mock_argv: MockArgvType
    ) -> None:
        with (
            mock_argv(
                "-n", "task", "--sandbox", "modal", "--sandbox-snapshot-name", "snap"
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2

    def test_snapshot_name_accepted_for_runloop(self, mock_argv: MockArgvType) -> None:
        with mock_argv(
            "-n", "task", "--sandbox", "runloop", "--sandbox-snapshot-name", "bp"
        ):
            parsed = parse_args()
            assert parsed.sandbox == "runloop"
            assert parsed.sandbox_snapshot_name == "bp"

    def test_sandbox_id_rejected_for_unsupported_provider(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reject `--sandbox-id` for agentcore (supports_sandbox_id=False)."""
        with (
            mock_argv("-n", "task", "--sandbox", "agentcore", "--sandbox-id", "abc"),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2
        assert "--sandbox-id is not supported" in capsys.readouterr().err

    def test_sandbox_id_accepted_for_supported_provider(
        self, mock_argv: MockArgvType
    ) -> None:
        with mock_argv("-n", "task", "--sandbox", "vercel", "--sandbox-id", "abc"):
            parsed = parse_args()
            assert parsed.sandbox == "vercel"
            assert parsed.sandbox_id == "abc"

    def test_snapshot_name_rejected_for_vercel(self, mock_argv: MockArgvType) -> None:
        with (
            mock_argv(
                "-n",
                "task",
                "--sandbox",
                "vercel",
                "--sandbox-snapshot-name",
                "snap",
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2

    def test_malformed_config_surfaces_note(
        self,
        mock_argv: MockArgvType,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A bare `--sandbox` against a malformed config explains the fault."""
        config = tmp_path / "config.toml"
        config.write_text("this is not = valid = toml", encoding="utf-8")
        with (
            patch(
                "deepagents_code.integrations.sandbox_config.DEFAULT_CONFIG_PATH",
                config,
            ),
            mock_argv("-n", "task", "--sandbox"),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2
        assert "could not be used" in capsys.readouterr().err


class TestNoStreamArgument:
    """Tests for --no-stream argument parsing."""

    def test_flag_stores_true(self, mock_argv: MockArgvType) -> None:
        """Test --no-stream sets no_stream to True."""
        with mock_argv("--no-stream", "-n", "task"):
            parsed = parse_args()
            assert parsed.no_stream is True

    def test_not_specified_is_false(self, mock_argv: MockArgvType) -> None:
        """Test no_stream is False when not provided."""
        with mock_argv():
            parsed = parse_args()
            assert parsed.no_stream is False

    def test_combined_with_quiet(self, mock_argv: MockArgvType) -> None:
        """Test --no-stream works alongside --quiet."""
        with mock_argv("--no-stream", "-q", "-n", "task"):
            parsed = parse_args()
            assert parsed.no_stream is True
            assert parsed.quiet is True

    def test_requires_non_interactive(self) -> None:
        """Test --no-stream without -n or piped stdin exits with code 2."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "--no-stream"]),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 2


class TestQuietRequiresNonInteractive:
    """Tests for --quiet validation in cli_main (after stdin pipe processing)."""

    def test_quiet_without_non_interactive_exits(self) -> None:
        """Test --quiet without -n or piped stdin exits with code 2."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "-q"]),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 2


class TestSkillFlagValidation:
    """Tests for `--skill` validation in `cli_main`."""

    def test_skill_allowed_with_non_interactive(self) -> None:
        """`--skill` should be accepted when `-n` selects headless mode."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "--skill", "code-review", "-n", "review this"],
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 0
        assert mock_run.await_args.kwargs["initial_skill"] == "code-review"  # ty: ignore

    def test_skill_with_quiet_without_non_interactive_exits_2(self) -> None:
        """`--skill` + `--quiet` without `-n` should exit with code 2."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "--skill", "code-review", "-q"],
            ),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 2

    def test_skill_with_no_stream_without_non_interactive_exits_2(self) -> None:
        """`--skill` + `--no-stream` without `-n` should exit with code 2."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "--skill", "code-review", "--no-stream"],
            ),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 2

    def test_skill_with_explicit_stdin_and_quiet_runs_headless(self) -> None:
        """`--skill --stdin -q` clears the guard and forwards the skill headless.

        Explicit `--stdin` routes the piped text to `non_interactive_message`
        (not the interactive `-m` seed), which satisfies the `--skill` +
        `--quiet` guard and reaches `run_non_interactive` with both the piped
        message and `initial_skill`.
        """
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "review this repo"
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "--skill", "code-review", "--stdin", "-q"],
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            # Skip the /dev/tty dance — os.open would fail in test sandboxes
            # and the real code path already tolerates that failure.
            patch("os.open", side_effect=OSError("No tty in test sandbox")),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 0
        assert mock_run.await_args.kwargs["initial_skill"] == "code-review"  # ty: ignore
        assert mock_run.await_args.kwargs["message"] == "review this repo"  # ty: ignore


class TestMaxTurnsArgument:
    """Tests for --max-turns argument parsing and validation."""

    def test_parses_integer(self, mock_argv: MockArgvType) -> None:
        """--max-turns N stores an integer."""
        with mock_argv("-n", "task", "--max-turns", "5"):
            parsed = parse_args()
            assert parsed.max_turns == 5

    def test_not_specified_is_none(self, mock_argv: MockArgvType) -> None:
        """max_turns is None when --max-turns is not provided."""
        with mock_argv():
            parsed = parse_args()
            assert parsed.max_turns is None

    def test_combined_with_non_interactive(self, mock_argv: MockArgvType) -> None:
        """--max-turns works alongside -n and other flags."""
        with mock_argv(
            "-n", "deploy app", "--max-turns", "10", "--shell-allow-list", "ls"
        ):
            parsed = parse_args()
            assert parsed.non_interactive_message == "deploy app"
            assert parsed.max_turns == 10
            assert parsed.shell_allow_list == "ls"

    def test_requires_non_interactive_mode(self) -> None:
        """--max-turns without -n or piped stdin exits with code 2."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "--max-turns", "5"]),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 2

    def test_allowed_with_piped_stdin(self) -> None:
        """--max-turns without -n is allowed when stdin is piped."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "piped task"
        with (
            patch.object(sys, "argv", ["deepagents", "--max-turns", "5"]),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            # Skip the /dev/tty dance — os.open would fail in test sandboxes
            # and the real code path already tolerates that failure.
            patch("os.open", side_effect=OSError("No tty in test sandbox")),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 0
        assert mock_run.await_args.kwargs["max_turns"] == 5  # ty: ignore

    def test_forwarded_to_run_non_interactive(self) -> None:
        """--max-turns value is forwarded to run_non_interactive as max_turns."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys, "argv", ["deepagents", "-n", "do the thing", "--max-turns", "3"]
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_run,
            pytest.raises(SystemExit),
        ):
            cli_main()
        assert mock_run.await_args.kwargs["max_turns"] == 3  # ty: ignore

    def test_not_forwarded_as_none_when_omitted(self) -> None:
        """When --max-turns is omitted, max_turns=None is forwarded."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "-n", "do the thing"]),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_run,
            pytest.raises(SystemExit),
        ):
            cli_main()
        assert mock_run.await_args.kwargs["max_turns"] is None  # ty: ignore

    @pytest.mark.parametrize("bad_value", ["0", "-1", "-50", "abc"])
    def test_rejects_non_positive_and_non_integer(
        self, mock_argv: MockArgvType, bad_value: str
    ) -> None:
        """Argparse rejects 0, negatives, and non-integers with exit 2."""
        with (
            mock_argv("-n", "task", "--max-turns", bad_value),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2


async def _raise_timeout_and_close(awaitable: object, **_kwargs: object) -> None:
    """Close the mocked awaitable before simulating a timeout."""
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()
    await asyncio.sleep(0)
    raise TimeoutError


def _wait_for_timeout(mock_wait_for: MagicMock) -> object:
    """Extract the `timeout` arg from a mocked `asyncio.wait_for` call.

    Handles both positional and keyword call styles so the assertion does not
    depend on how production code passes the argument.
    """
    import inspect

    call = mock_wait_for.call_args
    bound = inspect.signature(asyncio.wait_for).bind(*call.args, **call.kwargs)
    return bound.arguments["timeout"]


class TestTimeoutArgument:
    """Tests for --timeout argument parsing, validation, and runtime behavior."""

    def test_parses_integer(self, mock_argv: MockArgvType) -> None:
        """--timeout N stores an integer."""
        with mock_argv("-n", "task", "--timeout", "60"):
            parsed = parse_args()
            assert parsed.timeout == 60

    def test_not_specified_is_none(self, mock_argv: MockArgvType) -> None:
        """Timeout is None when --timeout is not provided."""
        with mock_argv():
            parsed = parse_args()
            assert parsed.timeout is None

    def test_combined_with_non_interactive(self, mock_argv: MockArgvType) -> None:
        """--timeout works alongside -n and --max-turns."""
        with mock_argv("-n", "run tests", "--timeout", "120", "--max-turns", "10"):
            parsed = parse_args()
            assert parsed.non_interactive_message == "run tests"
            assert parsed.timeout == 120
            assert parsed.max_turns == 10

    def test_requires_non_interactive_mode(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--timeout without -n or piped stdin exits with code 2 and warns on stderr."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "--timeout", "30"]),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 2
        stderr = capsys.readouterr().err
        assert "--timeout" in stderr
        assert "-n" in stderr

    def test_allowed_with_piped_stdin(self) -> None:
        """--timeout without -n is allowed when stdin is piped.

        Also asserts that `max_turns` (None by default) is still forwarded to
        `run_non_interactive`, guarding against kwarg drops in the surrounding
        try/except refactor.
        """
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "piped task"
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "--timeout", "30", "--max-turns", "5"],
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch("os.open", side_effect=OSError("No tty in test sandbox")),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 0
        mock_run.assert_awaited_once()
        await_args = mock_run.await_args
        assert await_args is not None
        assert await_args.kwargs["max_turns"] == 5

    def test_forwarded_via_wait_for(self) -> None:
        """--timeout value is used as the asyncio.wait_for timeout."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys, "argv", ["deepagents", "-n", "do the thing", "--timeout", "45"]
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch("asyncio.wait_for", wraps=asyncio.wait_for) as mock_wait_for,
            pytest.raises(SystemExit),
        ):
            cli_main()
        assert _wait_for_timeout(mock_wait_for) == 45

    def test_timeout_exits_124(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Exits with 124 and warns on stderr when `asyncio.TimeoutError` is raised."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys, "argv", ["deepagents", "-n", "slow task", "--timeout", "1"]
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "asyncio.wait_for",
                side_effect=_raise_timeout_and_close,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 124
        stderr = capsys.readouterr().err
        assert "timed out" in stderr
        assert "1s" in stderr

    def test_no_timeout_when_omitted(self) -> None:
        """When --timeout is omitted, wait_for is called with timeout=None."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "-n", "do the thing"]),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch("asyncio.wait_for", wraps=asyncio.wait_for) as mock_wait_for,
            pytest.raises(SystemExit),
        ):
            cli_main()
        assert _wait_for_timeout(mock_wait_for) is None

    @pytest.mark.parametrize("bad_value", ["0", "-1", "-60", "abc"])
    def test_rejects_non_positive_and_non_integer(
        self, mock_argv: MockArgvType, bad_value: str
    ) -> None:
        """Argparse rejects 0, negatives, and non-integers with exit 2."""
        with (
            mock_argv("-n", "task", "--timeout", bad_value),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2


class TestModelParamsArgument:
    """Tests for --model-params argument parsing."""

    def test_stores_json_string(self, mock_argv: MockArgvType) -> None:
        """Test --model-params stores the raw JSON string."""
        with mock_argv("--model-params", '{"temperature": 0.7}'):
            parsed = parse_args()
            assert parsed.model_params == '{"temperature": 0.7}'

    def test_not_specified_is_none(self, mock_argv: MockArgvType) -> None:
        """Test model_params is None when not provided."""
        with mock_argv():
            parsed = parse_args()
            assert parsed.model_params is None

    def test_combined_with_model(self, mock_argv: MockArgvType) -> None:
        """Test --model-params works alongside --model."""
        with mock_argv(
            "--model",
            "gpt-5.5",
            "--model-params",
            '{"temperature": 0.5, "max_tokens": 2048}',
        ):
            parsed = parse_args()
            assert parsed.model == "gpt-5.5"
            assert parsed.model_params == '{"temperature": 0.5, "max_tokens": 2048}'


class TestMaxRetriesForwarding:
    """`--max-retries` rides the forwarded model_params under an internal key.

    The value is carried under `CLI_MAX_RETRIES_KEY` rather than a literal
    `max_retries` so `create_model` can fold it under the resolved provider's
    retry-param name; see `TestRetriesConfig` in `test_config.py` for the
    folding/precedence behavior at the `create_model` layer.
    """

    def _run_model_params(self, argv: list[str]) -> dict[str, object] | None:
        """Drive `cli_main` and return the `model_params` passed downstream."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_run,
            pytest.raises(SystemExit),
        ):
            cli_main()
        await_args = mock_run.await_args
        assert await_args is not None
        return await_args.kwargs["model_params"]  # ty: ignore

    def test_folds_into_model_params(self) -> None:
        """`--max-retries` creates model_params when no `--model-params` is given."""
        from deepagents_code.config import CLI_MAX_RETRIES_KEY

        argv = ["deepagents", "-n", "task", "--max-retries", "4"]
        assert self._run_model_params(argv) == {CLI_MAX_RETRIES_KEY: 4}

    def test_carried_alongside_model_params(self) -> None:
        """`--max-retries` rides next to `--model-params` without clobbering it.

        The flag value is stashed under the internal key, leaving any explicit
        `--model-params` entries (including a literal `max_retries`) untouched in
        the forwarded dict. Precedence is resolved later, in `create_model`.
        """
        from deepagents_code.config import CLI_MAX_RETRIES_KEY

        argv = [
            "deepagents",
            "-n",
            "task",
            "--model-params",
            '{"max_retries": 1, "temperature": 0.5}',
            "--max-retries",
            "4",
        ]
        assert self._run_model_params(argv) == {
            "max_retries": 1,
            "temperature": 0.5,
            CLI_MAX_RETRIES_KEY: 4,
        }

    def test_absent_leaves_model_params_untouched(self) -> None:
        """Without `--max-retries`, model_params reflects only `--model-params`."""
        argv = ["deepagents", "-n", "task", "--model-params", '{"temperature": 0.5}']
        assert self._run_model_params(argv) == {"temperature": 0.5}


class TestProfileOverrideArgument:
    """Tests for --profile-override argument parsing."""

    def test_stores_json_string(self, mock_argv: MockArgvType) -> None:
        """--profile-override stores the raw JSON string."""
        with mock_argv("--profile-override", '{"max_input_tokens": 4096}'):
            parsed = parse_args()
            assert parsed.profile_override == '{"max_input_tokens": 4096}'

    def test_not_specified_is_none(self, mock_argv: MockArgvType) -> None:
        """profile_override is None when not provided."""
        with mock_argv():
            parsed = parse_args()
            assert parsed.profile_override is None

    def test_combined_with_model(self, mock_argv: MockArgvType) -> None:
        """--profile-override works alongside --model."""
        with mock_argv(
            "--model",
            "gpt-5.5",
            "--profile-override",
            '{"max_input_tokens": 4096}',
        ):
            parsed = parse_args()
            assert parsed.model == "gpt-5.5"
            assert parsed.profile_override == '{"max_input_tokens": 4096}'

    def test_invalid_json_exits(self) -> None:
        """--profile-override with invalid JSON exits with code 1."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "--profile-override", "{bad"]),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 1

    def test_non_dict_json_exits(self) -> None:
        """--profile-override with JSON array exits with code 1."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "--profile-override", "[1,2]"]),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 1


def _make_args(
    *,
    non_interactive_message: str | None = None,
    initial_prompt: str | None = None,
    initial_skill: str | None = None,
    stdin: bool = False,
) -> argparse.Namespace:
    """Create a minimal argument namespace for stdin pipe tests."""
    return argparse.Namespace(
        non_interactive_message=non_interactive_message,
        initial_prompt=initial_prompt,
        initial_skill=initial_skill,
        stdin=stdin,
    )


class TestApplyStdinPipe:
    """Tests for apply_stdin_pipe — reading piped stdin into CLI args."""

    def test_tty_is_noop(self) -> None:
        """When stdin is a TTY, args are not modified."""
        args = _make_args()
        with patch.object(sys, "stdin", wraps=sys.stdin) as mock_stdin:
            mock_stdin.isatty = lambda: True
            apply_stdin_pipe(args)
        assert args.non_interactive_message is None
        assert args.initial_prompt is None

    def test_empty_stdin_is_noop(self) -> None:
        """When piped stdin is empty/whitespace, args are not modified."""
        args = _make_args()
        fake_stdin = io.StringIO("   \n  ")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message is None
        assert args.initial_prompt is None

    def test_stdin_sets_non_interactive(self) -> None:
        """Piped stdin with no flags sets non_interactive_message."""
        args = _make_args()
        fake_stdin = io.StringIO("my prompt")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "my prompt"
        assert args.initial_prompt is None

    def test_stdin_prepends_to_non_interactive(self) -> None:
        """Piped stdin is prepended to an existing -n message."""
        args = _make_args(non_interactive_message="do something")
        fake_stdin = io.StringIO("context from pipe")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "context from pipe\n\ndo something"

    def test_stdin_prepends_to_initial_prompt(self) -> None:
        """Piped stdin is prepended to an existing -m message."""
        args = _make_args(initial_prompt="explain this")
        fake_stdin = io.StringIO("error log contents")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.initial_prompt == "error log contents\n\nexplain this"
        assert args.non_interactive_message is None

    def test_stdin_sets_initial_prompt_for_startup_skill(self) -> None:
        """Piped stdin becomes the startup request when `--skill` is set."""
        args = _make_args(initial_skill="code-review")
        fake_stdin = io.StringIO("diff contents")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.initial_prompt == "diff contents"
        assert args.non_interactive_message is None

    def test_stdin_prepends_to_skill_prompt(self) -> None:
        """Piped stdin is prepended when `--skill` and `-m` are combined."""
        args = _make_args(initial_prompt="review this", initial_skill="code-review")
        fake_stdin = io.StringIO("diff contents")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.initial_prompt == "diff contents\n\nreview this"
        assert args.non_interactive_message is None

    def test_explicit_stdin_with_skill_runs_headless(self) -> None:
        """Explicit `--stdin` + `--skill` runs headless, not the seeded TUI."""
        args = _make_args(initial_skill="code-review", stdin=True)
        fake_stdin = io.StringIO("review this repo")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "review this repo"
        assert args.initial_prompt is None

    def test_explicit_stdin_without_skill_sets_non_interactive(self) -> None:
        """Explicit `--stdin` with no skill/`-n`/`-m` sets non_interactive_message."""
        args = _make_args(stdin=True)
        fake_stdin = io.StringIO("my prompt")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "my prompt"
        assert args.initial_prompt is None

    def test_explicit_stdin_prepends_to_non_interactive(self) -> None:
        """Explicit `--stdin` still prepends to an existing -n message."""
        args = _make_args(non_interactive_message="do something", stdin=True)
        fake_stdin = io.StringIO("context from pipe")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "context from pipe\n\ndo something"
        assert args.initial_prompt is None

    def test_explicit_stdin_prepends_to_initial_prompt(self) -> None:
        """Explicit `--stdin` still merges into an existing -m message."""
        args = _make_args(initial_prompt="explain this", stdin=True)
        fake_stdin = io.StringIO("error log contents")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.initial_prompt == "error log contents\n\nexplain this"
        assert args.non_interactive_message is None

    def test_non_interactive_takes_priority_over_initial_prompt(self) -> None:
        """When both -n and -m are set, stdin is prepended to -n."""
        args = _make_args(non_interactive_message="task", initial_prompt="ignored")
        fake_stdin = io.StringIO("piped")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "piped\n\ntask"
        assert args.initial_prompt == "ignored"

    def test_multiline_stdin(self) -> None:
        """Multiline piped input is preserved."""
        args = _make_args()
        fake_stdin = io.StringIO("line one\nline two\nline three")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "line one\nline two\nline three"
        assert args.initial_prompt is None

    def test_none_stdin_is_noop(self) -> None:
        """When sys.stdin is None (embedded Python), args are not modified."""
        args = _make_args()
        with patch.object(sys, "stdin", None):
            apply_stdin_pipe(args)
        assert args.non_interactive_message is None
        assert args.initial_prompt is None

    def test_closed_stdin_is_noop(self) -> None:
        """When stdin.isatty() raises ValueError, treat as no pipe input."""
        args = _make_args()
        mock_stdin = MagicMock()
        mock_stdin.isatty.side_effect = ValueError("I/O operation on closed file")
        with patch.object(sys, "stdin", mock_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message is None
        assert args.initial_prompt is None

    def test_unicode_decode_error_exits(self) -> None:
        """Binary piped input triggers a clean exit, not a raw traceback."""
        args = _make_args()
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.side_effect = UnicodeDecodeError(
            "utf-8", b"\x80", 0, 1, "invalid start byte"
        )
        with (
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            apply_stdin_pipe(args)
        assert exc_info.value.code == 1

    def test_read_os_error_exits(self) -> None:
        """An OSError during stdin.read() exits with code 1."""
        args = _make_args()
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.side_effect = OSError("I/O error")
        with (
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            apply_stdin_pipe(args)
        assert exc_info.value.code == 1

    def test_read_value_error_exits(self) -> None:
        """A ValueError during stdin.read() exits with code 1."""
        args = _make_args()
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.side_effect = ValueError("I/O operation on closed file")
        with (
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            apply_stdin_pipe(args)
        assert exc_info.value.code == 1

    def test_oversized_stdin_exits(self) -> None:
        """Piped input exceeding the size limit triggers a clean exit."""
        args = _make_args()
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        # Return more bytes than the 10 MiB limit
        mock_stdin.read.return_value = "x" * (10 * 1024 * 1024 + 1)
        with (
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            apply_stdin_pipe(args)
        assert exc_info.value.code == 1

    def test_stdin_restores_tty(self) -> None:
        """After reading piped input, fd 0 is replaced with /dev/tty."""
        args = _make_args()
        fake_stdin = io.StringIO("hello")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with (
            patch.object(sys, "stdin", fake_stdin),
            patch("os.open", return_value=99) as mock_os_open,
            patch("os.dup2") as mock_dup2,
            patch("os.close") as mock_close,
            patch("builtins.open") as mock_open,
        ):
            apply_stdin_pipe(args)
        mock_os_open.assert_called_once_with("/dev/tty", os.O_RDONLY)
        mock_dup2.assert_called_once_with(99, 0)
        mock_close.assert_called_once_with(99)
        mock_open.assert_called_once_with(0, encoding="utf-8", closefd=False)

    def test_tty_open_failure_preserves_input(self) -> None:
        """When /dev/tty cannot be opened, piped input is still captured."""
        args = _make_args()
        fake_stdin = io.StringIO("hello")
        fake_stdin.isatty = lambda: False  # ty: ignore
        with (
            patch.object(sys, "stdin", fake_stdin),
            patch("os.open", side_effect=OSError("No controlling terminal")),
        ):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "hello"


class TestAgentResolutionScope:
    """Recent-agent fallback should only apply to session launches."""

    def test_threads_list_preserves_show_all_default(self) -> None:
        """Bare `threads list` must not inherit `[agents].recent` as a filter."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True

        with (
            patch.object(sys, "argv", ["deepagents", "threads", "list"]),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.model_config.load_recent_agent") as load_recent,
            patch("deepagents_code.main._recent_agent_is_valid") as valid_recent,
            patch(
                "deepagents_code.sessions.list_threads_command",
                new_callable=AsyncMock,
            ) as mock_list,
        ):
            cli_main()

        mock_list.assert_awaited_once()
        assert mock_list.await_args.kwargs["agent_name"] is None  # ty: ignore
        load_recent.assert_not_called()
        valid_recent.assert_not_called()


class TestThreadsListCwdFilter:
    """Tests for `deepagents threads list --cwd` path normalization."""

    @staticmethod
    def _run_threads_list(*args: str) -> AsyncMock:
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True

        with (
            patch.object(sys, "argv", ["deepagents", "threads", "list", *args]),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch(
                "deepagents_code.sessions.list_threads_command",
                new_callable=AsyncMock,
            ) as mock_list,
        ):
            cli_main()

        return mock_list

    def test_no_value_cwd_uses_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bare `--cwd` filters by the current working directory."""
        monkeypatch.chdir(tmp_path)

        mock_list = self._run_threads_list("--cwd")

        assert mock_list.await_args.kwargs["cwd"] == str(Path.cwd())  # ty: ignore

    def test_explicit_relative_cwd_is_normalized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit relative paths match absolute stored cwd metadata."""
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)

        mock_list = self._run_threads_list("--cwd", ".")

        assert mock_list.await_args.kwargs["cwd"] == str(project.resolve())  # ty: ignore

    def test_explicit_home_cwd_is_expanded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit `~` paths match absolute stored cwd metadata."""
        project = tmp_path / "repo"
        project.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))

        mock_list = self._run_threads_list("--cwd", "~/repo")

        assert mock_list.await_args.kwargs["cwd"] == str(project.resolve())  # ty: ignore


class TestResolveAgentArg:
    """Resolution order: explicit > -r fallback > recent > default."""

    @staticmethod
    def _args(**kwargs: object) -> argparse.Namespace:
        defaults = {"agent": None, "resume_thread": None}
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_explicit_agent_wins(self) -> None:
        """An explicit `-a <name>` bypasses default/recent lookup entirely."""
        from deepagents_code.main import _resolve_agent_arg

        with (
            patch("deepagents_code.model_config.load_default_agent") as load_default,
            patch("deepagents_code.model_config.load_recent_agent") as load_recent,
        ):
            assert _resolve_agent_arg(self._args(agent="coder")) == "coder"
            load_default.assert_not_called()
            load_recent.assert_not_called()

    def test_resume_thread_forces_default(self) -> None:
        """With -r present, default lets thread-metadata inference pick the agent."""
        from deepagents_code._constants import DEFAULT_AGENT_NAME
        from deepagents_code.main import _resolve_agent_arg

        with (
            patch(
                "deepagents_code.model_config.load_default_agent",
                return_value="coder",
            ) as load_default,
            patch(
                "deepagents_code.model_config.load_recent_agent",
                return_value="researcher",
            ) as load_recent,
        ):
            result = _resolve_agent_arg(self._args(resume_thread="abc123"))
            assert result == DEFAULT_AGENT_NAME
            load_default.assert_not_called()
            load_recent.assert_not_called()

    def test_default_takes_precedence_over_recent(self) -> None:
        """`[agents].default` (Ctrl+S in picker) wins over `[agents].recent`."""
        from deepagents_code.main import _resolve_agent_arg

        with (
            patch(
                "deepagents_code.model_config.load_default_agent",
                return_value="researcher",
            ),
            patch(
                "deepagents_code.model_config.load_recent_agent",
                return_value="coder",
            ),
            patch("deepagents_code.main._recent_agent_is_valid", return_value=True),
        ):
            assert _resolve_agent_arg(self._args()) == "researcher"

    def test_uses_recent_when_valid(self) -> None:
        """No -a, no -r, no default: use `[agents].recent` when the dir exists."""
        from deepagents_code.main import _resolve_agent_arg

        with (
            patch("deepagents_code.model_config.load_default_agent", return_value=None),
            patch(
                "deepagents_code.model_config.load_recent_agent",
                return_value="coder",
            ),
            patch("deepagents_code.main._recent_agent_is_valid", return_value=True),
        ):
            assert _resolve_agent_arg(self._args()) == "coder"

    def test_falls_back_when_default_missing_dir(self) -> None:
        """Stale `[agents].default` pointing at a deleted dir falls through."""
        from deepagents_code.main import _resolve_agent_arg

        # `_recent_agent_is_valid` is called for both default and recent;
        # return False for the default name, True for the recent name.
        def _validate(name: str) -> bool:
            return name == "coder"

        with (
            patch(
                "deepagents_code.model_config.load_default_agent",
                return_value="ghost",
            ),
            patch(
                "deepagents_code.model_config.load_recent_agent",
                return_value="coder",
            ),
            patch(
                "deepagents_code.main._recent_agent_is_valid",
                side_effect=_validate,
            ),
        ):
            assert _resolve_agent_arg(self._args()) == "coder"

    def test_falls_back_when_recent_missing_dir(self) -> None:
        """Stale `[agents].recent` pointing at a deleted dir falls through."""
        from deepagents_code._constants import DEFAULT_AGENT_NAME
        from deepagents_code.main import _resolve_agent_arg

        with (
            patch("deepagents_code.model_config.load_default_agent", return_value=None),
            patch(
                "deepagents_code.model_config.load_recent_agent",
                return_value="ghost",
            ),
            patch("deepagents_code.main._recent_agent_is_valid", return_value=False),
        ):
            assert _resolve_agent_arg(self._args()) == DEFAULT_AGENT_NAME

    def test_falls_back_when_no_recent(self) -> None:
        """No saved recent agent: final fallback is the hard-coded default."""
        from deepagents_code._constants import DEFAULT_AGENT_NAME
        from deepagents_code.main import _resolve_agent_arg

        with (
            patch("deepagents_code.model_config.load_default_agent", return_value=None),
            patch("deepagents_code.model_config.load_recent_agent", return_value=None),
        ):
            assert _resolve_agent_arg(self._args()) == DEFAULT_AGENT_NAME

    def test_falls_back_when_both_default_and_recent_stale(self) -> None:
        """Both keys point at deleted dirs → final fallback to DEFAULT_AGENT_NAME.

        Locks the chained validity check: a stale `default` does not
        suppress the `recent` lookup, but if `recent` is also stale we
        must reach `DEFAULT_AGENT_NAME` rather than returning a name
        whose directory no longer exists.
        """
        from deepagents_code._constants import DEFAULT_AGENT_NAME
        from deepagents_code.main import _resolve_agent_arg

        with (
            patch(
                "deepagents_code.model_config.load_default_agent",
                return_value="ghost-default",
            ),
            patch(
                "deepagents_code.model_config.load_recent_agent",
                return_value="ghost-recent",
            ),
            patch("deepagents_code.main._recent_agent_is_valid", return_value=False),
        ):
            assert _resolve_agent_arg(self._args()) == DEFAULT_AGENT_NAME


class TestRecentAgentIsValid:
    """`_recent_agent_is_valid` survives filesystem errors."""

    def test_returns_true_for_existing_dir(self, tmp_path, monkeypatch) -> None:
        """Existing `~/.deepagents/<name>/` resolves to True."""
        from deepagents_code.main import _recent_agent_is_valid

        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".deepagents" / "coder").mkdir(parents=True)

        assert _recent_agent_is_valid("coder") is True

    def test_returns_false_for_missing_dir(self, tmp_path, monkeypatch) -> None:
        """Missing dir → False, no exception."""
        from deepagents_code.main import _recent_agent_is_valid

        monkeypatch.setenv("HOME", str(tmp_path))

        assert _recent_agent_is_valid("ghost") is False

    def test_swallows_os_error(self) -> None:
        """A PermissionError or other OSError on is_dir() is logged and False."""
        from deepagents_code.main import _recent_agent_is_valid

        with patch("pathlib.Path.is_dir", side_effect=PermissionError("denied")):
            assert _recent_agent_is_valid("coder") is False


class TestUpdateSubcommand:
    """Control-flow tests for `deepagents update` and `--update`.

    Each branch has a destructive or user-visible failure mode (editable
    install would clobber a dev checkout; PyPI-unreachable must
    not be confused with up-to-date). These tests pin the dispatch order.
    """

    @staticmethod
    def _run_update(
        *,
        debug: bool = False,
        editable: bool,
        is_update_available_return: tuple[bool, str | None],
        log_path: str = "/tmp/deepagents-update.log",
        prerelease: bool = False,
        flag_style: bool = False,
        prerelease_before_command: bool = False,
        install_method: str = "uv",
        release_requires_prereleases: bool = False,
    ) -> tuple[int, MagicMock, MagicMock]:
        """Invoke `cli_main()` with `update`; return exit code + mocks."""
        from deepagents_code._env_vars import DEBUG_UPDATE
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        if prerelease_before_command:
            argv = ["deepagents", "--prerelease", "update"]
        elif flag_style:
            argv = ["deepagents", "--update"]
        else:
            argv = ["deepagents", "update"]
        if prerelease:
            argv.append("--prerelease")
        with (
            patch.dict(os.environ, {DEBUG_UPDATE: "1" if debug else ""}),
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.config._is_editable_install", return_value=editable),
            # `--prerelease` is only honored on uv installs; pin the detected
            # method so the precheck's outcome is driven by the test rather than
            # the test runner's own environment.
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value=install_method,
            ),
            patch(
                "deepagents_code.update_check.is_update_available",
                return_value=is_update_available_return,
            ) as is_update_mock,
            patch(
                "deepagents_code.update_check.release_requires_prereleases",
                return_value=release_requires_prereleases,
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=log_path,
            ),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_upgrade_mock,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        return int(exc_info.value.code or 0), is_update_mock, perform_upgrade_mock

    def test_editable_install_skips_upgrade(self) -> None:
        """Editable install exits 0 without calling `is_update_available`/upgrade.

        A regression here would run `uv tool upgrade deepagents-code` on an
        editable checkout and clobber the dev install with a PyPI copy.
        """
        code, is_update_mock, perform_upgrade_mock = self._run_update(
            editable=True,
            is_update_available_return=(True, "99.0.0"),
        )
        assert code == 0
        is_update_mock.assert_not_called()
        perform_upgrade_mock.assert_not_called()

    def test_pypi_unreachable_exits_nonzero(self) -> None:
        """`(False, None)` from `is_update_available` surfaces as exit 1."""
        code, _, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(False, None),
        )
        assert code == 1
        perform_upgrade_mock.assert_not_called()

    def test_up_to_date_exits_zero_without_upgrade(self) -> None:
        """`(False, "x.y.z")` exits 0 and does not call `perform_upgrade`."""
        code, _, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(False, "1.2.3"),
        )
        assert code == 0
        perform_upgrade_mock.assert_not_called()

    def test_debug_update_skips_upgrade(self) -> None:
        """Debug update mode exits 0 without invoking the installer."""
        code, _, perform_upgrade_mock = self._run_update(
            debug=True,
            editable=False,
            is_update_available_return=(True, "99.0.0"),
        )
        assert code == 0
        perform_upgrade_mock.assert_not_called()

    def test_markup_like_log_path_does_not_break_output(self) -> None:
        """Dynamic log paths are printed without Rich markup parsing."""
        code, _, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(True, "99.0.0"),
            log_path="/tmp/[/red]/deepagents-update.log",
        )
        assert code == 0
        perform_upgrade_mock.assert_awaited_once()

    def test_update_available_runs_upgrade(self) -> None:
        """`(True, "x.y.z")` triggers `perform_upgrade` and exits 0 on success."""
        code, is_update_mock, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(True, "99.0.0"),
        )
        assert code == 0
        is_update_mock.assert_called_once_with(
            bypass_cache=True,
            include_prereleases=None,
        )
        perform_upgrade_mock.assert_awaited_once_with(
            log_path="/tmp/deepagents-update.log",
            include_prereleases=None,
            target_version="99.0.0",
        )

    def test_stable_update_with_prerelease_deps_keeps_upgrade_intent_none(
        self,
    ) -> None:
        """Stable releases with pre-release deps let `perform_upgrade` pin the app."""
        code, is_update_mock, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(True, "99.0.0"),
            release_requires_prereleases=True,
        )

        assert code == 0
        is_update_mock.assert_called_once_with(
            bypass_cache=True,
            include_prereleases=None,
        )
        perform_upgrade_mock.assert_awaited_once_with(
            log_path="/tmp/deepagents-update.log",
            include_prereleases=None,
            target_version="99.0.0",
        )

    def test_prerelease_update_includes_prereleases(self) -> None:
        """`dcode update --prerelease` opts into alpha/beta/rc releases."""
        code, is_update_mock, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(True, "99.0.0rc1"),
            prerelease=True,
        )

        assert code == 0
        is_update_mock.assert_called_once_with(
            bypass_cache=True,
            include_prereleases=True,
        )
        perform_upgrade_mock.assert_awaited_once_with(
            log_path="/tmp/deepagents-update.log",
            include_prereleases=True,
            target_version="99.0.0rc1",
        )

    def test_prerelease_update_flag_includes_prereleases(self) -> None:
        """`dcode --update --prerelease` uses the same prerelease path."""
        code, is_update_mock, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(True, "99.0.0rc1"),
            prerelease=True,
            flag_style=True,
        )

        assert code == 0
        is_update_mock.assert_called_once_with(
            bypass_cache=True,
            include_prereleases=True,
        )
        perform_upgrade_mock.assert_awaited_once_with(
            log_path="/tmp/deepagents-update.log",
            include_prereleases=True,
            target_version="99.0.0rc1",
        )

    def test_prerelease_before_update_includes_prereleases(self) -> None:
        """`dcode --prerelease update` preserves the top-level prerelease flag."""
        code, is_update_mock, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(True, "99.0.0rc1"),
            prerelease_before_command=True,
        )

        assert code == 0
        is_update_mock.assert_called_once_with(
            bypass_cache=True,
            include_prereleases=True,
        )
        perform_upgrade_mock.assert_awaited_once_with(
            log_path="/tmp/deepagents-update.log",
            include_prereleases=True,
            target_version="99.0.0rc1",
        )

    def test_prerelease_unsupported_install_refuses_before_pypi(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--prerelease` on a non-uv install refuses before hitting PyPI.

        A regression that dropped the guard would let a brew/other install fall
        through to a stable update (or an unsupported upgrade attempt), so the
        refusal must short-circuit before `is_update_available`/`perform_upgrade`.
        """
        code, is_update_mock, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(True, "99.0.0rc1"),
            prerelease=True,
            install_method="brew",
        )

        assert code == 1
        is_update_mock.assert_not_called()
        perform_upgrade_mock.assert_not_called()
        captured = capsys.readouterr()
        assert "aren't supported for this install" in (captured.out + captured.err)

    def test_unexpected_error_manual_command_keeps_prerelease(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unexpected crash during `--prerelease` keeps the pre-release hint.

        The last-resort handler prints a manual upgrade command. A regression
        that hardcoded the stable command would nudge a user who requested a
        pre-release onto the stable channel — the exact silent downgrade this
        flag guards against, via an error side-door.
        """
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "update", "--prerelease"]),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            # Crash after the pre-release support check passes but before the
            # upgrade completes, exercising the catch-all fallback path.
            patch(
                "deepagents_code.update_check.is_update_available",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "--prerelease allow" in (captured.out + captured.err)

    def test_prerelease_without_update_exits_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--prerelease` only applies to the update command."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "--prerelease"]),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 2
        assert "--prerelease requires --update or the update subcommand" in (
            capsys.readouterr().err
        )


class TestInstallExtraSubcommand:
    """Control-flow tests for `dcode --install <extra>`."""

    @staticmethod
    def _run_install(
        extra: str,
        *,
        editable: bool = False,
        yes: bool = False,
        interactive: bool = False,
        perform_return: tuple[bool, str] = (True, ""),
        command_side_effect: BaseException | None = None,
    ) -> tuple[int, MagicMock]:
        """Invoke `cli_main()` with `--install`; return exit code + mock."""
        from deepagents_code.main import cli_main

        argv = ["deepagents", "--install", extra]
        if yes:
            argv.append("--yes")

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = interactive
        # Empty piped input so `apply_stdin_pipe` returns before its TTY
        # restoration path (`os.dup2`/`open(0)`) swaps out this mocked stdin
        # for a real terminal where `/dev/tty` is openable, which would mask
        # the handler's `isatty()` refusal check.
        mock_stdin.read.return_value = ""
        command_mock = MagicMock(
            return_value=(
                "curl -LsSf https://langch.in/dcode | "
                f"DEEPAGENTS_CODE_EXTRAS={extra} bash"
            ),
        )
        if command_side_effect is not None:
            command_mock.side_effect = command_side_effect
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            # `cli_main` resolves `console` via a lazy `__getattr__` on
            # `deepagents_code.config` that caches a single real `Console` in
            # the module globals for the whole worker process. Left unpatched,
            # the `--install` handler's `console.print(...)` calls run against
            # that shared instance, so console/stdout state leaked by an earlier
            # test in the same xdist worker can make `print` raise. The handler
            # wraps the flow in a broad `except Exception` that turns any such
            # error into `sys.exit(1)`, which would mask the intended refusal
            # exit code. Patch with `create=True` so the mock is installed
            # before the lazy import line runs.
            patch("deepagents_code.config.console", MagicMock(), create=True),
            patch("deepagents_code.config._is_editable_install", return_value=editable),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value="/tmp/deepagents-install.log",
            ),
            patch(
                "deepagents_code.update_check.install_extra_command",
                command_mock,
            ),
            patch(
                "deepagents_code.update_check.install_extra_recovery_command",
                command_mock,
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=perform_return,
            ) as perform_mock,
            patch("builtins.input", return_value="n"),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        return int(exc_info.value.code or 0), perform_mock

    def test_known_extra_runs_install(self) -> None:
        """A known extra invokes `perform_install_extra` and exits 0."""
        code, perform_mock = self._run_install("quickjs")
        assert code == 0
        perform_mock.assert_awaited_once()

    def test_editable_install_refuses(self) -> None:
        """Editable install short-circuits with a `uv sync` hint, exit 1."""
        code, perform_mock = self._run_install("quickjs", editable=True)
        assert code == 1
        perform_mock.assert_not_awaited()

    def test_unknown_extra_non_interactive_refuses(self) -> None:
        """Non-TTY stdin + unknown extra + no --yes must exit 2 (refusal)."""
        code, perform_mock = self._run_install("not-a-real-extra", interactive=False)
        assert code == 2
        perform_mock.assert_not_awaited()

    def test_invalid_extra_refuses_even_with_yes(self) -> None:
        """Malformed extras must never reach the installer command path."""
        code, perform_mock = self._run_install(
            "quickjs']; echo nope; '",
            yes=True,
            interactive=False,
        )
        assert code == 2
        perform_mock.assert_not_awaited()

    def test_unknown_extra_with_yes_runs(self) -> None:
        """`--yes` bypasses the unknown-extra confirmation."""
        code, perform_mock = self._run_install(
            "not-a-real-extra", yes=True, interactive=False
        )
        assert code == 0
        perform_mock.assert_awaited_once()

    @staticmethod
    def _run_install_capture(
        extra: str,
        *,
        editable: bool = False,
        yes: bool = False,
        interactive: bool = False,
        perform_return: tuple[bool, str] = (True, ""),
        perform_side_effect: BaseException | None = None,
        command_side_effect: BaseException | None = None,
        command_return: str | None = None,
        recovery_side_effect: BaseException | None = None,
        recovery_return: str | None = None,
        input_reply: str = "n",
    ) -> tuple[int, MagicMock, MagicMock]:
        """Invoke `cli_main()` with `--install` and capture console output.

        `install_extra_command` and `install_extra_recovery_command` share one
        mock by default (the realistic "both resolve identically" case). Pass
        `recovery_return` or `recovery_side_effect` to drive the recovery
        command independently — e.g. to exercise the path where the initial
        command resolves but the recovery command raises.

        Returns:
            `(exit_code, perform_mock, console_mock)` — *console_mock* is a
                `MagicMock` substituted for `deepagents_code.main.console`,
                so assertions can run against the recorded `.print(...)` calls.
        """
        from deepagents_code.main import cli_main

        argv = ["deepagents", "--install", extra]
        if yes:
            argv.append("--yes")

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = interactive
        # Empty piped input so `apply_stdin_pipe` returns before its TTY
        # restoration path clobbers this mocked stdin. See `_run_install`.
        mock_stdin.read.return_value = ""
        console_mock = MagicMock()
        perform_mock = AsyncMock()
        if perform_side_effect is not None:
            perform_mock.side_effect = perform_side_effect
        else:
            perform_mock.return_value = perform_return
        default_script_cmd = (
            f"curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_EXTRAS={extra} bash"
        )
        command_mock = MagicMock(return_value=command_return or default_script_cmd)
        if command_side_effect is not None:
            command_mock.side_effect = command_side_effect
        # Default: recovery resolves to the same command. Only split into its
        # own mock when a test drives the recovery command independently.
        if recovery_return is not None or recovery_side_effect is not None:
            recovery_mock = MagicMock(
                return_value=recovery_return or default_script_cmd
            )
            if recovery_side_effect is not None:
                recovery_mock.side_effect = recovery_side_effect
        else:
            recovery_mock = command_mock
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            # `cli_main` resolves `console` via a lazy `__getattr__` on
            # `deepagents_code.config`, so patch with `create=True` to
            # install the mock before the import line runs.
            patch("deepagents_code.config.console", console_mock, create=True),
            patch("deepagents_code.config._is_editable_install", return_value=editable),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/deepagents-install.log"),
            ),
            patch(
                "deepagents_code.update_check.install_extra_command",
                command_mock,
            ),
            patch(
                "deepagents_code.update_check.install_extra_recovery_command",
                recovery_mock,
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                perform_mock,
            ),
            patch("builtins.input", return_value=input_reply),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        return int(exc_info.value.code or 0), perform_mock, console_mock

    @staticmethod
    def _printed_text(console_mock: MagicMock) -> str:
        """Return the concatenated positional args of every `.print()` call."""
        chunks: list[str] = []
        for call in console_mock.print.call_args_list:
            chunks.extend(str(arg) for arg in call.args)
        return "\n".join(chunks)

    def test_success_renders_installed_message(self) -> None:
        """Successful install prints a green confirmation and exits 0."""
        code, _perform, console_mock = self._run_install_capture("quickjs")
        assert code == 0
        text = self._printed_text(console_mock)
        assert "Installed extra 'quickjs'" in text
        assert "tail -f /tmp/deepagents-install.log" in text

    def test_failure_renders_log_path_and_manual_command(self) -> None:
        """A failed install surfaces both the log path and manual script command."""
        code, _perform, console_mock = self._run_install_capture(
            "quickjs",
            perform_return=(False, "resolver: conflict"),
        )
        assert code == 1
        text = self._printed_text(console_mock)
        assert "Install failed" in text
        assert "resolver: conflict" in text
        assert "/tmp/deepagents-install.log" in text
        assert "curl -LsSf https://langch.in/dcode" in text
        assert "DEEPAGENTS_CODE_EXTRAS=quickjs bash" in text
        assert "quickjs" in text

    def test_failure_escapes_uv_recovery_command_markup(self) -> None:
        """Failed uv recovery commands preserve extras rendered by Rich."""
        command = "uv tool install -U 'deepagents-code[quickjs]'"
        code, _perform, console_mock = self._run_install_capture(
            "quickjs",
            perform_return=(False, "resolver: conflict"),
            command_return=command,
        )
        assert code == 1
        text = self._printed_text(console_mock)
        assert "deepagents-code\\[quickjs]" in text

    def test_failure_recovery_command_error_keeps_prior_command(self) -> None:
        """A recovery-command error on a failed install keeps the prior command.

        The command resolved before the failure is shown instead of crashing.
        """
        resolved = "uv tool install -U 'deepagents-code[quickjs]'"
        code, _perform, console_mock = self._run_install_capture(
            "quickjs",
            perform_return=(False, "resolver: conflict"),
            command_return=resolved,
            recovery_side_effect=ValueError("bad receipt"),
        )
        assert code == 1
        text = self._printed_text(console_mock)
        assert "Install failed" in text
        # Falls back to the install_extra_command value resolved before the
        # failure, with its bracket escaped for Rich.
        assert "deepagents-code\\[quickjs]" in text

    def test_keyboard_interrupt_exits_130(self) -> None:
        """Ctrl-C during install exits 130 with an Aborted message."""
        code, _perform, console_mock = self._run_install_capture(
            "quickjs",
            perform_side_effect=KeyboardInterrupt(),
        )
        assert code == 130
        assert "Aborted" in self._printed_text(console_mock)

    def test_unexpected_exception_includes_class_and_log(self) -> None:
        """Outer except prints the exception class, message, and log path."""
        code, _perform, console_mock = self._run_install_capture(
            "quickjs",
            perform_side_effect=RuntimeError("disk full"),
        )
        assert code == 1
        text = self._printed_text(console_mock)
        assert "RuntimeError" in text
        assert "disk full" in text
        assert "/tmp/deepagents-install.log" in text
        assert "curl -LsSf https://langch.in/dcode" in text
        assert "DEEPAGENTS_CODE_EXTRAS=quickjs bash" in text
        assert "quickjs" in text

    def test_command_generation_exception_uses_literal_fallback(self) -> None:
        """If resolved command construction fails, the fallback command is shown."""
        code, perform_mock, console_mock = self._run_install_capture(
            "quickjs",
            command_side_effect=RuntimeError("metadata broken"),
        )
        assert code == 1
        perform_mock.assert_not_awaited()
        text = self._printed_text(console_mock)
        assert "RuntimeError" in text
        assert "metadata broken" in text
        assert "Run manually: " in text
        assert "curl -LsSf https://langch.in/dcode" in text
        assert "DEEPAGENTS_CODE_EXTRAS=quickjs bash" in text

    def test_interactive_decline_aborts(self) -> None:
        """Interactive TTY + reply 'n' to unknown extra aborts with exit 1."""
        code, perform_mock, console_mock = self._run_install_capture(
            "not-a-real-extra",
            interactive=True,
            input_reply="n",
        )
        assert code == 1
        perform_mock.assert_not_awaited()
        assert "Aborted" in self._printed_text(console_mock)


class TestInstallPackageSubcommand:
    """Control-flow tests for `dcode --install <pkg> --package`."""

    @staticmethod
    def _run_install_package(
        package: str,
        *,
        with_install: bool = True,
        editable: bool = False,
        yes: bool = False,
        interactive: bool = False,
        perform_return: tuple[bool, str] = (True, ""),
        perform_side_effect: BaseException | None = None,
        input_reply: str = "n",
    ) -> tuple[int, MagicMock, MagicMock]:
        """Invoke `cli_main()` with `--package`; return exit code + mocks."""
        from deepagents_code.main import cli_main

        argv = ["deepagents"]
        if with_install:
            argv += ["--install", package]
        argv.append("--package")
        if yes:
            argv.append("--yes")

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = interactive
        # Empty piped input so `apply_stdin_pipe` returns before its TTY
        # restoration path clobbers this mocked stdin. See `_run_install`.
        mock_stdin.read.return_value = ""
        console_mock = MagicMock()
        if perform_side_effect is not None:
            perform_mock = AsyncMock(side_effect=perform_side_effect)
        else:
            perform_mock = AsyncMock(return_value=perform_return)
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch("deepagents_code.config.console", console_mock, create=True),
            patch("deepagents_code.config._is_editable_install", return_value=editable),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/deepagents-install.log"),
            ),
            patch(
                "deepagents_code.update_check.perform_install_package",
                perform_mock,
            ),
            patch("builtins.input", return_value=input_reply),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        return int(exc_info.value.code or 0), perform_mock, console_mock

    @staticmethod
    def _printed_text(console_mock: MagicMock) -> str:
        chunks: list[str] = []
        for call in console_mock.print.call_args_list:
            chunks.extend(str(arg) for arg in call.args)
        return "\n".join(chunks)

    def test_package_with_yes_runs(self) -> None:
        """`--package --yes` invokes `perform_install_package` and exits 0."""
        code, perform_mock, console_mock = self._run_install_package(
            "langchain-custom", yes=True, interactive=False
        )
        assert code == 0
        perform_mock.assert_awaited_once()
        text = self._printed_text(console_mock)
        assert "Installed package 'langchain-custom'" in text
        assert "tail -f /tmp/deepagents-install.log" in text

    def test_package_non_interactive_without_yes_refuses(self) -> None:
        """Non-TTY stdin + no --yes must exit 2 without installing."""
        code, perform_mock, _console = self._run_install_package(
            "langchain-custom", interactive=False
        )
        assert code == 2
        perform_mock.assert_not_awaited()

    def test_package_editable_install_refuses(self) -> None:
        """Editable install short-circuits with a `--with` hint, exit 1."""
        code, perform_mock, _console = self._run_install_package(
            "langchain-custom", editable=True, yes=True
        )
        assert code == 1
        perform_mock.assert_not_awaited()

    def test_invalid_package_refuses_even_with_yes(self) -> None:
        """Malformed package names must never reach the installer command path."""
        code, perform_mock, _console = self._run_install_package(
            "custom;touch", yes=True, interactive=False
        )
        assert code == 2
        perform_mock.assert_not_awaited()

    def test_package_flag_without_install_errors(self) -> None:
        """`--package` with no `--install` value must exit 2."""
        code, perform_mock, _console = self._run_install_package(
            "langchain-custom", with_install=False
        )
        assert code == 2
        perform_mock.assert_not_awaited()

    def test_package_interactive_decline_aborts(self) -> None:
        """Interactive TTY + reply 'n' aborts with exit 1."""
        code, perform_mock, console_mock = self._run_install_package(
            "langchain-custom", interactive=True, input_reply="n"
        )
        assert code == 1
        perform_mock.assert_not_awaited()
        assert "Aborted" in self._printed_text(console_mock)

    def test_package_failure_renders_log(self) -> None:
        """A failed package install surfaces the detail + log path, no uv command."""
        code, _perform, console_mock = self._run_install_package(
            "langchain-custom", yes=True, perform_return=(False, "resolver: conflict")
        )
        assert code == 1
        text = self._printed_text(console_mock)
        assert "Install failed" in text
        assert "resolver: conflict" in text
        assert "/tmp/deepagents-install.log" in text
        # The raw `uv tool` command is never surfaced to the user.
        assert "uv tool" not in text

    def test_package_keyboard_interrupt_exits_130(self) -> None:
        """Ctrl+C during the install exits 130 via the catch-all."""
        code, _perform, console_mock = self._run_install_package(
            "langchain-custom", yes=True, perform_side_effect=KeyboardInterrupt()
        )
        assert code == 130
        assert "Aborted" in self._printed_text(console_mock)

    def test_package_unexpected_error_exits_nonzero_with_log(self) -> None:
        """An unexpected exception is caught, logged, and exits 1 (not a traceback)."""
        code, _perform, console_mock = self._run_install_package(
            "langchain-custom", yes=True, perform_side_effect=RuntimeError("boom")
        )
        assert code == 1
        text = self._printed_text(console_mock)
        assert "RuntimeError" in text
        assert "uv tool" not in text

    def test_option_injection_name_refused(self) -> None:
        """A leading-dash name is rejected before any install path (exit 2)."""
        code, perform_mock, _console = self._run_install_package("-rreqs.txt", yes=True)
        assert code == 2
        perform_mock.assert_not_awaited()


class TestParseInterpreterToolsFlag:
    """Tests for `_parse_interpreter_tools_flag`."""

    def test_none_returns_none(self) -> None:
        from deepagents_code.main import _parse_interpreter_tools_flag

        assert _parse_interpreter_tools_flag(None) is None

    def test_safe_sentinel(self) -> None:
        from deepagents_code.main import _parse_interpreter_tools_flag

        assert _parse_interpreter_tools_flag("safe") == "safe"

    def test_all_sentinel(self) -> None:
        from deepagents_code.main import _parse_interpreter_tools_flag

        assert _parse_interpreter_tools_flag("all") == "all"

    def test_explicit_list(self) -> None:
        from deepagents_code.main import _parse_interpreter_tools_flag

        assert _parse_interpreter_tools_flag("read_file,glob,grep,task") == [
            "read_file",
            "glob",
            "grep",
            "task",
        ]

    def test_safe_inside_list(self) -> None:
        from deepagents_code.main import _parse_interpreter_tools_flag

        assert _parse_interpreter_tools_flag("safe,task") == ["safe", "task"]

    def test_all_inside_list_exits(self) -> None:
        from deepagents_code.main import _parse_interpreter_tools_flag

        with pytest.raises(SystemExit) as exc_info:
            _parse_interpreter_tools_flag("all,task")
        assert exc_info.value.code == 2

    def test_empty_value_exits(self) -> None:
        from deepagents_code.main import _parse_interpreter_tools_flag

        with pytest.raises(SystemExit) as exc_info:
            _parse_interpreter_tools_flag("   ")
        assert exc_info.value.code == 2


class TestInterpreterFlagParsing:
    """`--interpreter` is a tri-state `BooleanOptionalAction` (default `None`)."""

    def test_default_is_none(self, mock_argv: MockArgvType) -> None:
        with mock_argv("-n", "task"):
            args = parse_args()
        assert args.interpreter is None

    def test_explicit_flag_is_true(self, mock_argv: MockArgvType) -> None:
        with mock_argv("-n", "task", "--interpreter"):
            args = parse_args()
        assert args.interpreter is True

    def test_no_flag_is_false(self, mock_argv: MockArgvType) -> None:
        with mock_argv("-n", "task", "--no-interpreter"):
            args = parse_args()
        assert args.interpreter is False


class TestResolveInterpreterEnabled:
    """Tests for `_resolve_interpreter_enabled`."""

    def test_local_mode_uses_config_default(self, mock_argv: MockArgvType) -> None:
        from deepagents_code.config import settings
        from deepagents_code.main import _resolve_interpreter_enabled

        with mock_argv("-n", "task"):
            args = parse_args()
        with patch.object(settings, "enable_interpreter", False):
            assert _resolve_interpreter_enabled(args) is False
        with patch.object(settings, "enable_interpreter", True):
            assert _resolve_interpreter_enabled(args) is True

    def test_sandbox_defaults_disabled(self, mock_argv: MockArgvType) -> None:
        from deepagents_code.config import settings
        from deepagents_code.main import _resolve_interpreter_enabled

        with mock_argv("-n", "task", "--sandbox", "daytona"):
            args = parse_args()
        with patch.object(settings, "enable_interpreter", True):
            assert _resolve_interpreter_enabled(args) is False

    def test_explicit_flag_overrides_sandbox_default(
        self, mock_argv: MockArgvType
    ) -> None:
        from deepagents_code.main import _resolve_interpreter_enabled

        with mock_argv("-n", "task", "--sandbox", "daytona", "--interpreter"):
            args = parse_args()
        assert _resolve_interpreter_enabled(args) is True

    def test_explicit_no_flag_overrides_local_default(
        self, mock_argv: MockArgvType
    ) -> None:
        from deepagents_code.config import settings
        from deepagents_code.main import _resolve_interpreter_enabled

        with mock_argv("-n", "task", "--no-interpreter"):
            args = parse_args()
        with patch.object(settings, "enable_interpreter", True):
            assert _resolve_interpreter_enabled(args) is False

    def test_empty_sandbox_treated_as_local(self) -> None:
        """An empty-string sandbox is falsy and must resolve as local mode.

        Parity with the `_resolve_enable_interpreter` edge case (the CLI resolver
        delegates to it); a regression in the falsy-sandbox guard must fail here
        too, not only at the server-config layer.
        """
        import argparse

        from deepagents_code.config import settings
        from deepagents_code.main import _resolve_interpreter_enabled

        args = argparse.Namespace(interpreter=None, sandbox="")
        with patch.object(settings, "enable_interpreter", True):
            assert _resolve_interpreter_enabled(args) is True


class TestRunTextualCliAsyncInterpreterDefault:
    """Tests for TUI helper interpreter tri-state forwarding."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            (True, True),
            (False, False),
        ],
    )
    async def test_forwards_interpreter_tri_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: bool | None,
        expected: bool | None,
    ) -> None:
        from deepagents_code.app import AppResult
        from deepagents_code.main import run_textual_cli_async

        run_textual_app = AsyncMock(
            return_value=AppResult(return_code=0, thread_id="thread")
        )
        monkeypatch.setattr(
            "deepagents_code.config._get_default_model_spec",
            lambda: "test-model",
        )
        monkeypatch.setattr("deepagents_code.config.detect_provider", lambda _: "")
        monkeypatch.setattr(
            "deepagents_code.onboarding.should_run_onboarding",
            lambda: False,
        )
        monkeypatch.setattr("deepagents_code.app.run_textual_app", run_textual_app)

        if value is not None:
            await run_textual_cli_async(
                assistant_id="agent",
                sandbox_type="daytona",
                enable_interpreter=value,
            )
        else:
            await run_textual_cli_async(
                assistant_id="agent",
                sandbox_type="daytona",
            )

        server_kwargs = run_textual_app.call_args.kwargs["server_kwargs"]
        assert server_kwargs["enable_interpreter"] is expected


class TestWarnInterpreterToolsWithoutInterpreter:
    """Tests for `_warn_if_interpreter_tools_without_interpreter`."""

    def test_warns_when_interpreter_disabled(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--interpreter-tools with --no-interpreter warns and does not exit."""
        from deepagents_code.main import (
            _warn_if_interpreter_tools_without_interpreter,
        )

        with mock_argv("-n", "task", "--no-interpreter", "--interpreter-tools", "safe"):
            args = parse_args()
        _warn_if_interpreter_tools_without_interpreter(args, enable_interpreter=False)
        assert (
            "--interpreter-tools has no effect when the interpreter is disabled"
            in capsys.readouterr().err
        )

    def test_no_warning_with_interpreter(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--interpreter --interpreter-tools does not warn."""
        from deepagents_code.main import (
            _warn_if_interpreter_tools_without_interpreter,
        )

        with mock_argv("-n", "task", "--interpreter", "--interpreter-tools", "safe"):
            args = parse_args()
        _warn_if_interpreter_tools_without_interpreter(args, enable_interpreter=True)
        assert capsys.readouterr().err == ""

    def test_no_warning_without_flag(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Absent --interpreter-tools does not warn."""
        from deepagents_code.main import (
            _warn_if_interpreter_tools_without_interpreter,
        )

        with mock_argv("-n", "task"):
            args = parse_args()
        _warn_if_interpreter_tools_without_interpreter(args, enable_interpreter=True)
        assert capsys.readouterr().err == ""

    def test_cli_main_emits_warning_on_non_interactive_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end: `-n --no-interpreter --interpreter-tools` warns.

        Guards the wiring (not just the helper): a dropped or misplaced call
        site, or an earlier `sys.exit` swallowing the warning, fails here.
        """
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys,
                "argv",
                [
                    "deepagents",
                    "-n",
                    "task",
                    "--no-interpreter",
                    "--interpreter-tools",
                    "safe",
                ],
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive",
                new_callable=AsyncMock,
                return_value=0,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        assert (
            "--interpreter-tools has no effect when the interpreter is disabled"
            in capsys.readouterr().err
        )


class TestWarnInterpreterDisabledBySandbox:
    """Tests for `_warn_if_interpreter_disabled_by_sandbox` (stderr advisory)."""

    def test_warns_when_sandbox_suppresses_default(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A `--sandbox` run with the default-on interpreter warns on stderr."""
        from deepagents_code.config import settings
        from deepagents_code.main import _warn_if_interpreter_disabled_by_sandbox

        with mock_argv("-n", "task", "--sandbox", "daytona"):
            args = parse_args()
        with patch.object(settings, "enable_interpreter", True):
            _warn_if_interpreter_disabled_by_sandbox(args)
        assert "unavailable under a remote sandbox" in capsys.readouterr().err

    def test_silent_in_local_mode(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Local mode keeps the interpreter, so there is nothing to warn about."""
        from deepagents_code.config import settings
        from deepagents_code.main import _warn_if_interpreter_disabled_by_sandbox

        with mock_argv("-n", "task"):
            args = parse_args()
        with patch.object(settings, "enable_interpreter", True):
            _warn_if_interpreter_disabled_by_sandbox(args)
        assert capsys.readouterr().err == ""

    def test_silent_on_explicit_opt_out_under_sandbox(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An explicit `--no-interpreter` is the user's choice, not a drop."""
        from deepagents_code.config import settings
        from deepagents_code.main import _warn_if_interpreter_disabled_by_sandbox

        with mock_argv("-n", "task", "--sandbox", "daytona", "--no-interpreter"):
            args = parse_args()
        with patch.object(settings, "enable_interpreter", True):
            _warn_if_interpreter_disabled_by_sandbox(args)
        assert capsys.readouterr().err == ""

    def test_silent_when_config_default_off(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A user who disabled the interpreter in config is not nagged."""
        from deepagents_code.config import settings
        from deepagents_code.main import _warn_if_interpreter_disabled_by_sandbox

        with mock_argv("-n", "task", "--sandbox", "daytona"):
            args = parse_args()
        with patch.object(settings, "enable_interpreter", False):
            _warn_if_interpreter_disabled_by_sandbox(args)
        assert capsys.readouterr().err == ""

    def test_cli_main_forwards_enabled_interpreter_in_local_mode(self) -> None:
        """End-to-end: bare `-n task` resolves the interpreter on and forwards it.

        Guards the `cli_main` -> `run_non_interactive` wiring: a dropped or
        reverted assignment (e.g. back to a hard-coded `False`) fails here.
        """
        from deepagents_code.config import settings
        from deepagents_code.main import cli_main

        run_mock = AsyncMock(return_value=0)
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "-n", "task"]),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch.object(settings, "enable_interpreter", True),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive", run_mock
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        assert run_mock.call_args.kwargs["enable_interpreter"] is True

    def test_cli_main_warns_and_disables_under_sandbox(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end: `-n --sandbox` forwards the disabled interpreter and warns."""
        from deepagents_code.config import settings
        from deepagents_code.main import cli_main

        run_mock = AsyncMock(return_value=0)
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys, "argv", ["deepagents", "-n", "task", "--sandbox", "daytona"]
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.integrations.sandbox_factory.verify_sandbox_deps",
            ),
            patch.object(settings, "enable_interpreter", True),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive", run_mock
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        assert run_mock.call_args.kwargs["enable_interpreter"] is False
        assert "unavailable under a remote sandbox" in capsys.readouterr().err

    def test_cli_main_silent_on_explicit_opt_out_under_sandbox(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end: `-n --sandbox --no-interpreter` disables without an advisory."""
        from deepagents_code.config import settings
        from deepagents_code.main import cli_main

        run_mock = AsyncMock(return_value=0)
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys,
                "argv",
                [
                    "deepagents",
                    "-n",
                    "task",
                    "--sandbox",
                    "daytona",
                    "--no-interpreter",
                ],
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.integrations.sandbox_factory.verify_sandbox_deps",
            ),
            patch.object(settings, "enable_interpreter", True),
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive", run_mock
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        assert run_mock.call_args.kwargs["enable_interpreter"] is False
        assert "unavailable under a remote sandbox" not in capsys.readouterr().err
