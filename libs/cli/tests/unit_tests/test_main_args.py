"""Tests for command-line argument parsing."""

import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from unittest.mock import patch

import pytest

from deepagents_cli.config import parse_shell_allow_list
from deepagents_cli.main import parse_args

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
