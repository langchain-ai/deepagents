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
