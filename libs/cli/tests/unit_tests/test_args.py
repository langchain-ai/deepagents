"""Tests for CLI argument parsing."""

import sys
from unittest.mock import patch

from deepagents_cli.main import parse_args


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
