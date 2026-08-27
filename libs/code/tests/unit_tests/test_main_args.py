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


def test_malformed_shell_allow_list_is_a_visible_cli_error(
    mock_argv: MockArgvType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid explicit policy must abort instead of falling through."""
    with mock_argv("--shell-allow-list", "all,ls"), pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "--shell-allow-list" in error
    assert "Cannot combine 'all' with other commands" in error


@pytest.mark.parametrize("value", ["", "   ", ",", " , , "])
def test_empty_shell_allow_list_is_a_visible_cli_error(
    value: str,
    mock_argv: MockArgvType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit allow-list must contain at least one command."""
    with mock_argv("--shell-allow-list", value), pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "--shell-allow-list" in error
    assert "must contain at least one non-empty command" in error


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


class TestResolveApprovalMode:
    """Tests for `_resolve_approval_mode` (flag vs. `[startup].mode`)."""

    def test_blocked_recent_auto_queues_an_explanation(self, tmp_path: Path) -> None:
        """A stale Auto notice fails closed and explains the Manual fallback."""
        from deepagents_code.configuration import service
        from deepagents_code.main import _resolve_approval_mode
        from deepagents_code.model_config import (
            consume_recent_auto_not_restored_notice,
        )

        (tmp_path / "config.toml").write_text(
            '[startup]\nrecent = "auto"\n', encoding="utf-8"
        )
        consume_recent_auto_not_restored_notice()
        service.invalidate_config_sources()
        try:
            args = argparse.Namespace(auto_approve=None, yolo=False)
            assert _resolve_approval_mode(args).value == "manual"
            assert consume_recent_auto_not_restored_notice() is not None
        finally:
            service.invalidate_config_sources()

    def test_removed_dangerously_auto_spelling_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """The retired `dangerously-auto` spelling must not grant autonomy.

        Previously asserted by patching `model_config.load_startup_mode`, which
        `_resolve_approval_mode` stopped calling when it moved to the resolver.
        The patch became a no-op and the assertion passed on the empty-config
        default instead -- green whether or not the value fell back safely.
        Write the real spelling into `config.toml` so the coercion runs.
        """
        from deepagents_code.configuration import service
        from deepagents_code.main import _resolve_approval_mode

        (tmp_path / "config.toml").write_text(
            '[startup]\nmode = "dangerously-auto"\n', encoding="utf-8"
        )
        service.invalidate_config_sources()
        try:
            args = argparse.Namespace(auto_approve=None, yolo=False)
            assert _resolve_approval_mode(args).value == "manual"
        finally:
            service.invalidate_config_sources()


class TestYoloAcknowledgement:
    """Tests for the versioned local unrestricted-mode acknowledgement."""

    def test_new_acknowledgement_must_persist(self) -> None:
        from deepagents_code.main import _ensure_yolo_acknowledged

        console = MagicMock()
        with (
            patch(
                "deepagents_code.approval_mode.has_yolo_acknowledgement",
                return_value=False,
            ),
            patch(
                "deepagents_code.main._prompt_yolo_acknowledgement",
                return_value=True,
            ),
            patch(
                "deepagents_code.approval_mode.save_yolo_acknowledgement",
                return_value=False,
            ),
        ):
            assert not _ensure_yolo_acknowledged(console)
        assert console.print.called


class TestHeadlessApprovalFlagHandling:
    """Headless handling of the approval flags, which is deliberately split.

    `--auto-approve`/`--yolo` are ignored with a warning, because the same
    command line is commonly reused interactive and headless. Its dependent
    `--auto-classifier-model` still exits 2 — it has no interactive-reuse case,
    so a silent no-op there would only hide a typo. Both dispositions live here.
    """

    @pytest.mark.parametrize(
        ("argv", "piped_stdin", "flag", "managed_toml"),
        [
            (
                ["deepagents", "-y", "-n", "do the thing"],
                None,
                "--auto-approve",
                None,
            ),
            (
                ["deepagents", "--auto-approve"],
                "do the thing",
                "--auto-approve",
                None,
            ),
            (["deepagents", "--yolo", "-n", "task"], None, "--yolo", None),
            (
                ["deepagents", "--auto-approve", "-n", "task"],
                None,
                "--auto-approve",
                '[startup]\nmode = "manual"\n',
            ),
            (
                ["deepagents", "--yolo", "-n", "task"],
                None,
                "--yolo",
                '[startup]\nmode = "manual"\n',
            ),
        ],
        ids=[
            "explicit-headless",
            "piped-stdin",
            "yolo",
            "managed-manual-auto-approve",
            "managed-manual-yolo",
        ],
    )
    def test_headless_mode_ignores_interactive_approval_flag_with_warning(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
        piped_stdin: str | None,
        flag: str,
        managed_toml: str | None,
    ) -> None:
        """The run must complete (exit 0), not abort as it did before.

        `_resolve_interpreter_enabled` is patched as a probe, snapshotting the
        flags *during* the call, so the assertion pins that they are cleared
        before mode dispatch rather than merely by the time `cli_main` returns
        (`call_args` holds a live reference to the mutated namespace, so a
        post-dispatch clear would still look green).

        The managed-policy cases cover the flag the user typed surviving
        `_apply_managed_runtime_exceptions` revoking it: the warning keys off a
        parse-time capture, so it must still fire.
        """
        from deepagents_code.main import cli_main

        if managed_toml is not None:
            from deepagents_code.configuration import service
            from unit_tests.conftest import redirect_managed_config

            managed = tmp_path / "managed.toml"
            managed.write_text(managed_toml, encoding="utf-8")
            redirect_managed_config(monkeypatch, managed)
            service.invalidate_config_sources()

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = piped_stdin is None
        mock_stdin.read.return_value = piped_stdin
        seen: dict[str, object] = {}
        resolve_interpreter = MagicMock(
            side_effect=lambda ns: (
                seen.update(auto_approve=ns.auto_approve, yolo=ns.yolo) or False
            )
        )
        # Scoped to `/dev/tty`: a blanket `os.open` failure also disables the
        # fail-closed guard in `_prepare_debug_file`, which then floods the
        # stderr this test asserts on (reachable whenever DEEPAGENTS_CODE_DEBUG
        # is set, e.g. from a developer's ~/.deepagents/.env).
        real_open = os.open
        no_tty = OSError("No controlling terminal")

        def _open_no_tty(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if os.fsdecode(path) == "/dev/tty":
                raise no_tty
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", mock_stdin),
            patch("os.open", side_effect=_open_no_tty),
            patch("deepagents_code.main.check_optional_tools", return_value=[]),
            patch(
                "deepagents_code.main._should_ensure_managed_ripgrep",
                return_value=False,
            ),
            patch(
                "deepagents_code.main._resolve_interpreter_enabled",
                resolve_interpreter,
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
        raw = capsys.readouterr().err
        # Asserted on the *raw* stream, not whitespace-normalized: with the
        # pre-existing `sys.exit(2)` gone this line is the only signal a CI job
        # has, so it must stay greppable. Rich hard wraps at width 80 off a TTY
        # and the break moves with the flag name, so a normalized assertion
        # would hide a regression in the `soft_wrap` that prevents it.
        warning = next(
            (line for line in raw.splitlines() if "has no effect" in line), None
        )
        assert warning is not None, raw
        assert (
            warning == f"Warning: {flag} has no effect in headless mode; ignoring it. "
            "Shell access is governed by --shell-allow-list, and MCP routing "
            "is fail-closed."
        )
        assert seen == {"auto_approve": False, "yolo": False}

    def test_classifier_rejection_precedes_the_approval_flag_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A fatal classifier rejection must not be preceded by the warning.

        `-y --auto-classifier-model X -n ...` used to print "--auto-approve has
        no effect" and *then* exit 2 about the classifier flag — two verdicts
        for one command line. The classifier guard now runs first, so the exit
        is the only thing the user sees.
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
                    "-y",
                    "--auto-classifier-model",
                    "anthropic:claude-haiku-4-5",
                    "-n",
                    "task",
                ],
            ),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 2
        stderr = capsys.readouterr().err
        assert "--auto-classifier-model is only supported" in stderr
        assert "has no effect in headless mode" not in stderr

    def test_rejects_auto_classifier_model_with_sandbox(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Auto is disabled under a sandbox, so its classifier flag is a no-op.

        `create_cli_agent` turns Auto off for a sandboxed run, so accepting the
        flag would silently ignore a setting that governs action authorization.
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
                    "--sandbox",
                    "daytona",
                    "--auto-classifier-model",
                    "anthropic:claude-haiku-4-5",
                ],
            ),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 2
        assert "--auto-classifier-model is only supported" in capsys.readouterr().err

    def test_rejects_auto_classifier_model_when_headless(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A headless run has no Auto mode, so its classifier flag is a no-op.

        The mirror of the sandbox case, and the likelier user mistake. Without
        the `args.non_interactive_message` conjunct in the guard, `dcode -n ...
        --auto-classifier-model X` silently accepts a setting that governs action
        authorization and then discards it.
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
                    "do something",
                    "--auto-classifier-model",
                    "anthropic:claude-haiku-4-5",
                ],
            ),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--auto-classifier-model is only supported" in err
        assert "it runs headlessly" in err

    def test_accepts_auto_approve_in_interactive_mode(self) -> None:
        """`--auto-approve` must still be honored on an interactive launch.

        The guard clears the approval flags only when
        `args.non_interactive_message` is also set. Without that conjunct it
        would wrongly ignore `dcode -m ... -y`; this pins the interactive path
        so a dropped conjunct fails loudly instead of silently breaking the
        flag's primary use. Also asserts the resolved value flows through to
        the TUI (`auto_approve=True`).
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
        from deepagents_code.approval_mode import ApprovalMode

        assert await_args.kwargs["approval_mode"] is ApprovalMode.AUTO

    def test_auto_approve_downgraded_to_manual_with_sandbox(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--auto-approve` with a sandbox must downgrade to Manual and warn.

        Auto's classifier runs only in the sandbox-free local TUI. When a
        sandbox is requested the interactive launch path (`cli_main`) must
        resolve Manual and surface the reason instead of silently dropping the
        requested mode.
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
            patch.object(
                sys,
                "argv",
                ["deepagents", "--auto-approve", "--sandbox", "daytona", "-m", "hi"],
            ),
            patch.object(sys, "stdin", mock_stdin),
            # Skip the real provider dependency check; it exits before the
            # approval-mode downgrade when `daytona` extras are absent.
            patch(
                "deepagents_code.integrations.sandbox_factory.verify_sandbox_deps",
                return_value=None,
            ),
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
        from deepagents_code.approval_mode import ApprovalMode

        assert await_args.kwargs["approval_mode"] is ApprovalMode.MANUAL
        assert "Auto is unavailable with a sandbox" in capsys.readouterr().out


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


class TestQuietRequiresNonInteractive:
    """Tests for --quiet validation in cli_main (after stdin pipe processing)."""


class TestSkillFlagValidation:
    """Tests for `--skill` validation in `cli_main`."""

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

    def test_combined_with_non_interactive(self, mock_argv: MockArgvType) -> None:
        """--max-turns works alongside -n and other flags."""
        with mock_argv(
            "-n", "deploy app", "--max-turns", "10", "--shell-allow-list", "ls"
        ):
            parsed = parse_args()
            assert parsed.non_interactive_message == "deploy app"
            assert parsed.max_turns == 10
            assert parsed.shell_allow_list == "ls"

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


def test_cli_main_installs_the_shell_allow_list_before_dispatch() -> None:
    """`--shell-allow-list` must be resolvable by the time the agent is built.

    The flag's whole effect rests on an ordering invariant:
    `_install_cli_provider` must run before the work that reads the value.
    Nothing else fails if that ordering breaks -- the flag still parses, no
    warning fires, and shell auto-approval simply stops applying.

    Asserted against the resolver rather than the process singleton so the
    test observes the CLI provider installed by this invocation.

    Scope, verified by mutation: this fails if the manifest binding names the
    wrong argparse destination, and it does *not* fail if the explicit
    `_install_cli_provider` call is deleted -- `_resolver_for_args` installs a
    provider on demand, so the two paths are redundant. That redundancy is why
    the flag survives; do not read a pass here as proof the explicit call is
    still in place.
    """
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.resolver import (
        CLI_RANK,
        get_config_resolver,
    )
    from deepagents_code.main import cli_main

    seen: dict[str, object] = {}

    async def _capture(**_kwargs: object) -> int:  # noqa: RUF029  # awaited by cli_main
        option = get_option("shell.allow_list")
        assert option is not None
        resolved = get_config_resolver().get(option)
        seen["value"] = resolved.value
        seen["ranks"] = resolved.ranks
        return 0

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    with (
        patch.object(
            sys,
            "argv",
            ["deepagents", "-n", "task", "--shell-allow-list", "git status,ls"],
        ),
        patch.object(sys, "stdin", mock_stdin),
        patch("deepagents_code.main.check_optional_tools", return_value=[]),
        patch(
            "deepagents_code.main._should_ensure_managed_ripgrep",
            return_value=False,
        ),
        patch(
            "deepagents_code.client.non_interactive.run_non_interactive",
            _capture,
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    assert seen["value"] == ["git status", "ls"]
    assert seen["ranks"] == (CLI_RANK,)


class TestTimeoutArgument:
    """Tests for --timeout argument parsing, validation, and runtime behavior."""


class TestModelParamsArgument:
    """Tests for --model-params argument parsing."""


class TestMaxRetriesForwarding:
    """`--max-retries` stays separate from provider model parameters."""

    def _run_model_kwargs(self, argv: list[str]) -> dict[str, object]:
        """Drive `cli_main` and return model-related downstream arguments."""
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
        return {
            "model_params": await_args.kwargs["model_params"],
            "cli_max_retries": await_args.kwargs["cli_max_retries"],
        }


class TestProfileOverrideArgument:
    """Tests for --profile-override argument parsing."""


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


class TestAgentResolutionScope:
    """Recent-agent fallback should only apply to session launches."""


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


class TestResolveAgentArg:
    """Resolution order: explicit > -r fallback > recent > default."""

    @staticmethod
    def _args(**kwargs: object) -> argparse.Namespace:
        defaults = {"agent": None, "resume_thread": None}
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

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


class TestRecentAgentIsValid:
    """`_recent_agent_is_valid` survives filesystem errors."""

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
        # Runs in place of the stubbed `perform_upgrade`, for assertions about
        # state that only holds mid-install.
        upgrade_side_effect: Callable[..., object] | None = None,
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
                "deepagents_code.update_check.create_update_log_file",
                return_value=log_path,
            ),
            patch(
                "deepagents_code.update_check.perform_upgrade",
                new_callable=AsyncMock,
                return_value=(True, "", None),
                side_effect=upgrade_side_effect,
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

    def test_update_skips_install_while_another_process_holds_lock(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Headless updates must not race another process's install."""
        from deepagents_code.update_check import update_install_lock

        with update_install_lock() as holding:
            assert holding is True
            code, _, perform_upgrade_mock = self._run_update(
                editable=False,
                is_update_available_return=(True, "99.0.0"),
            )

        assert code == 1
        perform_upgrade_mock.assert_not_awaited()
        assert "Another dcode session is currently updating" in capsys.readouterr().out

    def test_update_runs_install_while_holding_the_lock(self) -> None:
        """The install must run inside the lock, not merely after a check.

        Shrinking the `with` block to cover only the boolean check would leave
        the install unguarded while the deferral test above still passed. The
        lock is not reentrant, so re-entering from inside `perform_upgrade`
        proves it is held for the real work.
        """
        from deepagents_code.update_check import update_install_lock

        held_during_install: list[bool] = []

        def _record_lock_state(**_kwargs: object) -> tuple[bool, str, None]:
            with update_install_lock() as holding:
                held_during_install.append(holding)
            return True, "", None

        code, _, perform_upgrade_mock = self._run_update(
            editable=False,
            is_update_available_return=(True, "99.0.0"),
            upgrade_side_effect=_record_lock_state,
        )

        assert code == 0
        perform_upgrade_mock.assert_awaited_once()
        assert held_during_install == [False], (
            "the install ran without holding the update lock"
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


class TestParseAllowFsToolsFlag:
    """Tests for `_parse_allow_fs_tools_flag`."""


class TestAllowFsToolsArgument:
    """Tests for --allow-fs-tools argument parsing and forwarding."""

    def test_invalid_value_exits_before_startup_side_effects(self) -> None:
        """Malformed allowlists fail before migration, installs, or prompts."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "-n", "task", "--allow-fs-tools", "bogus"],
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch("deepagents_code.state_migration.migrate_legacy_state") as migrate,
            patch("deepagents_code.main.check_optional_tools") as check_tools,
            patch("deepagents_code.main._run_startup_auto_update") as update,
            patch("deepagents_code.main._check_mcp_project_trust") as trust,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 2
        migrate.assert_not_called()
        check_tools.assert_not_called()
        update.assert_not_called()
        trust.assert_not_called()

    def test_forwarded_to_run_textual_cli(self) -> None:
        """--allow-fs-tools is parsed and forwarded to the TUI launch path."""
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
            patch.object(
                sys,
                "argv",
                ["deepagents", "-m", "hello", "--allow-fs-tools", "ls,read_file"],
            ),
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
        assert run_tui.await_args is not None
        assert run_tui.await_args.kwargs["allow_fs_tools"] == ["ls", "read_file"]


class TestInterpreterFlagParsing:
    """`--interpreter` is a tri-state `BooleanOptionalAction` (default `None`)."""


class TestResolveInterpreterEnabled:
    """Tests for `_resolve_interpreter_enabled`."""

    def test_explicit_flag_with_remote_sandbox_is_a_visible_error(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--interpreter --sandbox <remote>` must abort with a real message.

        The combination is unsatisfiable: `agent.py` refuses to build a remote
        sandbox with the interpreter enabled. Returning `True` here surfaced
        that as a bare `ValueError` traceback from deep inside agent
        construction, which tells the user nothing about which two flags
        conflict.
        """
        from deepagents_code.main import _resolve_interpreter_enabled

        with mock_argv("-n", "task", "--sandbox", "daytona", "--interpreter"):
            args = parse_args()
        with pytest.raises(SystemExit) as exc_info:
            _resolve_interpreter_enabled(args)
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "daytona" in out
        assert "--interpreter" in out

    def test_managed_false_revokes_an_explicit_interpreter_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Managed `false` must beat `--interpreter`, an enforced key.

        This branch removed the assignment that used to enforce
        `interpreter.enable_interpreter` in `_apply_managed_runtime_policy`;
        only `models.default` kept its copy. Enforcement now rests entirely on
        `deciding_rank` preferring `MANAGED_RANK`, and nothing covered the
        revocation direction -- a CLI-wins short-circuit inserted before
        `deciding_rank` left the whole suite green while a user's flag turned
        on JS execution against policy.
        """
        import argparse

        from deepagents_code.configuration import service
        from deepagents_code.main import _resolve_interpreter_enabled
        from unit_tests.conftest import redirect_managed_config

        managed = tmp_path / "managed.toml"
        managed.write_text(
            "[interpreter]\nenable_interpreter = false\n", encoding="utf-8"
        )
        redirect_managed_config(monkeypatch, managed)
        service.invalidate_config_sources()
        try:
            args = argparse.Namespace(interpreter=True, sandbox=None)
            assert _resolve_interpreter_enabled(args) is False
        finally:
            service.invalidate_config_sources()

    def test_managed_true_survives_an_explicit_opt_out(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`--no-interpreter` must not revoke a managed `true` either.

        The mirror of the revocation case: an enforced key is enforced in both
        directions, so neither spelling of the flag may override policy.
        """
        import argparse

        from deepagents_code.configuration import service
        from deepagents_code.main import _resolve_interpreter_enabled
        from unit_tests.conftest import redirect_managed_config

        managed = tmp_path / "managed.toml"
        managed.write_text(
            "[interpreter]\nenable_interpreter = true\n", encoding="utf-8"
        )
        redirect_managed_config(monkeypatch, managed)
        service.invalidate_config_sources()
        try:
            args = argparse.Namespace(interpreter=False, sandbox=None)
            assert _resolve_interpreter_enabled(args) is True
        finally:
            service.invalidate_config_sources()

    def test_managed_policy_outranks_the_sandbox_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Managed `enable_interpreter` must not be silently dropped.

        Regression: only the CLI tier was consulted, so a managed `true` --
        an `ENFORCED_MANAGED_KEYS` member -- silently became `false` whenever
        `--sandbox` named a remote backend. Policy may not be revoked by a
        user's flag, and the combination cannot be honoured either, so the
        launch stops and names the administrator as the way out.
        """
        import argparse

        from deepagents_code.configuration import service
        from deepagents_code.main import _resolve_interpreter_enabled
        from unit_tests.conftest import redirect_managed_config

        managed = tmp_path / "managed.toml"
        managed.write_text(
            "[interpreter]\nenable_interpreter = true\n", encoding="utf-8"
        )
        redirect_managed_config(monkeypatch, managed)
        service.invalidate_config_sources()
        try:
            args = argparse.Namespace(interpreter=None, sandbox="daytona")
            with pytest.raises(SystemExit) as exc_info:
                _resolve_interpreter_enabled(args)
            assert exc_info.value.code == 1
            out = capsys.readouterr().out
            assert "administrator" in out
        finally:
            service.invalidate_config_sources()

    def test_non_strict_reports_absent_instead_of_exiting(
        self, mock_argv: MockArgvType
    ) -> None:
        """`dcode tools` must report, not abort, on an unsatisfiable pair.

        A read-only listing has nothing to abort, and the interpreter would in
        fact be absent, so `strict=False` reports `False`.
        """
        from deepagents_code.main import _resolve_interpreter_enabled

        with mock_argv("-n", "task", "--sandbox", "daytona", "--interpreter"):
            args = parse_args()
        assert _resolve_interpreter_enabled(args, strict=False) is False

    @pytest.mark.parametrize(
        "toml_text",
        [
            "[interpreter]\nenable_interpreter = true\n",
            "[interpreter]\nenable_interpreter = false\n",
        ],
        ids=["true", "false"],
    )
    def test_user_config_does_not_override_the_sandbox_rule(
        self,
        tmp_path: Path,
        toml_text: str,
    ) -> None:
        """`config.toml` is an ambient preference; `--sandbox` is about this run.

        Regression: treating *any* declaring tier as decisive meant the
        redundant-but-harmless `enable_interpreter = true` in a user's
        `config.toml` resolved `True` under a remote sandbox, which
        `agent.py` then rejected with a `ValueError`. Selecting a remote
        sandbox used to just work for these users and must keep working.
        """
        import argparse

        from deepagents_code.configuration import service
        from deepagents_code.main import _resolve_interpreter_enabled

        # `_isolate_state_dir` already points `DEFAULT_CONFIG_PATH` here.
        (tmp_path / "config.toml").write_text(toml_text, encoding="utf-8")
        service.invalidate_config_sources()
        try:
            args = argparse.Namespace(interpreter=None, sandbox="daytona")
            assert _resolve_interpreter_enabled(args) is False
        finally:
            service.invalidate_config_sources()

    def test_sandbox_default_still_applies_when_undeclared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With policy silent on the key, the sandbox default stands."""
        import argparse

        from deepagents_code.configuration import service
        from deepagents_code.main import _resolve_interpreter_enabled
        from unit_tests.conftest import redirect_managed_config

        managed = tmp_path / "managed.toml"
        managed.write_text("[runtime]\nrecursion_limit = 100\n", encoding="utf-8")
        redirect_managed_config(monkeypatch, managed)
        service.invalidate_config_sources()
        try:
            args = argparse.Namespace(interpreter=None, sandbox="daytona")
            assert _resolve_interpreter_enabled(args) is False
        finally:
            service.invalidate_config_sources()


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


class TestWarnInterpreterDisabledBySandbox:
    """Tests for `_warn_if_interpreter_disabled_by_sandbox` (stderr advisory)."""

    def test_warns_when_sandbox_suppresses_default(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A `--sandbox` run with the default-on interpreter warns on stderr."""
        from deepagents_code.main import _warn_if_interpreter_disabled_by_sandbox

        with mock_argv("-n", "task", "--sandbox", "daytona"):
            args = parse_args()
        _warn_if_interpreter_disabled_by_sandbox(args)
        assert "unavailable under a remote sandbox" in capsys.readouterr().err

    def test_silent_in_local_mode(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Local mode keeps the interpreter, so there is nothing to warn about."""
        from deepagents_code.main import _warn_if_interpreter_disabled_by_sandbox

        with mock_argv("-n", "task"):
            args = parse_args()
        _warn_if_interpreter_disabled_by_sandbox(args)
        assert capsys.readouterr().err == ""

    def test_silent_on_explicit_opt_out_under_sandbox(
        self, mock_argv: MockArgvType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An explicit `--no-interpreter` is the user's choice, not a drop."""
        from deepagents_code.main import _warn_if_interpreter_disabled_by_sandbox

        with mock_argv("-n", "task", "--sandbox", "daytona", "--no-interpreter"):
            args = parse_args()
        _warn_if_interpreter_disabled_by_sandbox(args)
        assert capsys.readouterr().err == ""

    def test_silent_when_config_default_off(
        self,
        mock_argv: MockArgvType,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """A user who disabled the interpreter in config is not nagged."""
        from deepagents_code.configuration import service
        from deepagents_code.main import _warn_if_interpreter_disabled_by_sandbox

        (tmp_path / "config.toml").write_text(
            "[interpreter]\nenable_interpreter = false\n", encoding="utf-8"
        )
        service.invalidate_config_sources()
        with mock_argv("-n", "task", "--sandbox", "daytona"):
            args = parse_args()
        _warn_if_interpreter_disabled_by_sandbox(args)
        assert capsys.readouterr().err == ""

    def test_cli_main_warns_and_disables_under_sandbox(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end: `-n --sandbox` forwards the disabled interpreter and warns."""
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
            patch(
                "deepagents_code.client.non_interactive.run_non_interactive", run_mock
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        assert run_mock.call_args.kwargs["enable_interpreter"] is False
        assert "unavailable under a remote sandbox" not in capsys.readouterr().err


class TestModelParamsRetryOverrideWarning:
    """An ignored `--model-params` retry count must be visible, not buffered."""

    @staticmethod
    def _run_headless(argv: list[str]) -> None:
        """Run `cli_main` headlessly so the caller can read captured stderr."""
        from deepagents_code.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "hi"

        real_open = os.open

        def _open_no_tty(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if os.fsdecode(path) == "/dev/tty":
                msg = "No controlling terminal"
                raise OSError(msg)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", mock_stdin),
            patch("os.open", side_effect=_open_no_tty),
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
            pytest.raises(SystemExit),
        ):
            cli_main()

    def test_supplied_retry_param_warns(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--model-params max_retries` is always overridden, so say so."""
        # Pin the provider: without `--model` it resolves from available
        # credentials, and on a credential-free machine the provider is
        # unknown, so no retry kwarg is forced and the warning never fires.
        self._run_headless(
            [
                "dcode",
                "--model",
                "anthropic:claude-opus-4-5",
                "--model-params",
                '{"max_retries": 10}',
                "-n",
                "hi",
            ]
        )
        stderr = capsys.readouterr().err
        assert "--model-params max_retries is ignored" in stderr
        assert "--max-retries" in stderr
        # Rich parses `[retries]` as a style tag and drops it, which would
        # delete the remediation this warning exists to deliver.
        assert "[retries].max_retries in config.toml" in stderr

    def test_retry_config_warning_renders_markup_as_text(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Malformed config values cannot inject Rich markup or crash startup."""
        warning = "Ignoring [retries].max_retries='[/bold]'; expected an integer."
        with patch(
            "deepagents_code.config.collect_retry_config_startup",
            return_value=([warning], set()),
        ):
            self._run_headless(["dcode", "-n", "hi"])

        assert warning in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv_extra", "expected"),
    [
        pytest.param([], False, id="default-off"),
        pytest.param(["--show-reasoning"], True, id="flag-on"),
    ],
)
def test_cli_main_forwards_show_reasoning_to_headless(
    argv_extra: list[str], expected: bool
) -> None:
    """`--show-reasoning` must reach the headless runner, not just resolve.

    The flag resolving correctly proves nothing on its own: `cli_main` reads the
    preference and passes it as a separate kwarg, and dropping that kwarg leaves
    the flag parsing, the manifest binding intact, and the feature dead.
    """
    from deepagents_code.main import cli_main

    run_mock = AsyncMock(return_value=0)
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    with (
        patch.object(sys, "argv", ["deepagents", "-n", "task", *argv_extra]),
        patch.object(sys, "stdin", mock_stdin),
        patch("deepagents_code.main.check_optional_tools", return_value=[]),
        patch(
            "deepagents_code.main._should_ensure_managed_ripgrep",
            return_value=False,
        ),
        patch("deepagents_code.client.non_interactive.run_non_interactive", run_mock),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    assert run_mock.await_args.kwargs["show_reasoning"] is expected  # ty: ignore


class TestSummarizationModelForwarding:
    """The dedicated summary model reaches every headless input route."""

    def test_blank_flag_overrides_the_configured_default(self) -> None:
        """`--summarization-model ""` means "use the main model this launch".

        The resolver tests `is not None`, so an explicitly blank flag outranks
        `[models].summarization_default` -- the same escape hatch
        `--auto-classifier-model` documents.
        """
        from deepagents_code.main import cli_main
        from deepagents_code.model_config import ModelConfig

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        config = ModelConfig(summarization_default_model="openai:config-summary")
        with (
            patch.object(
                sys,
                "argv",
                ["deepagents", "-n", "task", "--summarization-model", ""],
            ),
            patch.object(sys, "stdin", mock_stdin),
            patch.object(ModelConfig, "load", return_value=config),
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
        assert not mock_run.await_args.kwargs["summarization_model"]  # ty: ignore

    def test_cli_override_reaches_explicit_headless_mode(self) -> None:
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
                    "--summarization-model",
                    "openai:summary-model",
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
            ) as mock_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()

        assert exc_info.value.code == 0
        assert (
            mock_run.await_args.kwargs["summarization_model"]  # ty: ignore
            == "openai:summary-model"
        )

    def test_config_default_reaches_piped_stdin(self) -> None:
        from deepagents_code.main import cli_main
        from deepagents_code.model_config import ModelConfig

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "piped task"
        config = ModelConfig(summarization_default_model="openai:config-summary")
        with (
            patch.object(sys, "argv", ["deepagents"]),
            patch.object(sys, "stdin", mock_stdin),
            patch.object(ModelConfig, "load", return_value=config),
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
        assert (
            mock_run.await_args.kwargs["summarization_model"]  # ty: ignore
            == "openai:config-summary"
        )

    def test_config_default_reaches_the_tui(self) -> None:
        """Every launch mode gets the same resolved spec, the TUI included."""
        from deepagents_code.main import cli_main
        from deepagents_code.model_config import ModelConfig

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        config = ModelConfig(summarization_default_model="openai:config-summary")

        fake_result = MagicMock()
        fake_result.return_code = 0
        fake_result.thread_id = None
        fake_result.update_available = (False, None)
        fake_result.session_stats = MagicMock(request_count=0)
        run_tui = AsyncMock(return_value=fake_result)

        with (
            patch.object(sys, "argv", ["deepagents", "-m", "hello"]),
            patch.object(sys, "stdin", mock_stdin),
            patch.object(ModelConfig, "load", return_value=config),
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
        assert run_tui.await_args is not None
        assert (
            run_tui.await_args.kwargs["summarization_model"] == "openai:config-summary"
        )
