"""Tests for command-line argument parsing."""

import argparse
import io
import os
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch

import pytest

from deepagents_cli.config import parse_shell_allow_list
from deepagents_cli.main import apply_stdin_pipe, parse_args

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


def test_shell_allow_list_combined_with_other_args(mock_argv: MockArgvType) -> None:
    """Test that shell-allow-list works with other arguments."""
    with mock_argv(
        "--shell-allow-list", "ls,cat", "--model", "gpt-4o", "--auto-approve"
    ):
        parsed_args = parse_args()
        assert parsed_args.shell_allow_list == "ls,cat"
        assert parsed_args.model == "gpt-4o"
        assert parsed_args.auto_approve is True


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
        from deepagents_cli.main import cli_main

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
        from deepagents_cli.main import cli_main

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["deepagents", "-q"]),
            patch.object(sys, "stdin", mock_stdin),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
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
            "gpt-4o",
            "--model-params",
            '{"temperature": 0.5, "max_tokens": 2048}',
        ):
            parsed = parse_args()
            assert parsed.model == "gpt-4o"
            assert parsed.model_params == '{"temperature": 0.5, "max_tokens": 2048}'


def _make_args(
    *,
    non_interactive_message: str | None = None,
    initial_prompt: str | None = None,
) -> argparse.Namespace:
    """Create a minimal argument namespace for stdin pipe tests."""
    return argparse.Namespace(
        non_interactive_message=non_interactive_message,
        initial_prompt=initial_prompt,
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
        fake_stdin.isatty = lambda: False  # type: ignore[attr-defined]
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message is None
        assert args.initial_prompt is None

    def test_stdin_sets_non_interactive(self) -> None:
        """Piped stdin with no flags sets non_interactive_message."""
        args = _make_args()
        fake_stdin = io.StringIO("my prompt")
        fake_stdin.isatty = lambda: False  # type: ignore[attr-defined]
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "my prompt"
        assert args.initial_prompt is None

    def test_stdin_prepends_to_non_interactive(self) -> None:
        """Piped stdin is prepended to an existing -n message."""
        args = _make_args(non_interactive_message="do something")
        fake_stdin = io.StringIO("context from pipe")
        fake_stdin.isatty = lambda: False  # type: ignore[attr-defined]
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "context from pipe\n\ndo something"

    def test_stdin_prepends_to_initial_prompt(self) -> None:
        """Piped stdin is prepended to an existing -m message."""
        args = _make_args(initial_prompt="explain this")
        fake_stdin = io.StringIO("error log contents")
        fake_stdin.isatty = lambda: False  # type: ignore[attr-defined]
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.initial_prompt == "error log contents\n\nexplain this"
        assert args.non_interactive_message is None

    def test_non_interactive_takes_priority_over_initial_prompt(self) -> None:
        """When both -n and -m are set, stdin is prepended to -n."""
        args = _make_args(non_interactive_message="task", initial_prompt="ignored")
        fake_stdin = io.StringIO("piped")
        fake_stdin.isatty = lambda: False  # type: ignore[attr-defined]
        with patch.object(sys, "stdin", fake_stdin):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "piped\n\ntask"
        assert args.initial_prompt == "ignored"

    def test_multiline_stdin(self) -> None:
        """Multiline piped input is preserved."""
        args = _make_args()
        fake_stdin = io.StringIO("line one\nline two\nline three")
        fake_stdin.isatty = lambda: False  # type: ignore[attr-defined]
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
        fake_stdin.isatty = lambda: False  # type: ignore[attr-defined]
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
        fake_stdin.isatty = lambda: False  # type: ignore[attr-defined]
        with (
            patch.object(sys, "stdin", fake_stdin),
            patch("os.open", side_effect=OSError("No controlling terminal")),
        ):
            apply_stdin_pipe(args)
        assert args.non_interactive_message == "hello"
