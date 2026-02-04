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


class TestThreadsExportArg:
    """Tests for threads export subcommand arguments."""

    def test_export_with_thread_id(self) -> None:
        """Verify thread_id is parsed correctly."""
        with patch.object(sys, "argv", ["deepagents", "threads", "export", "abc123"]):
            args = parse_args()
        assert args.command == "threads"
        assert args.threads_command == "export"
        assert args.thread_id == "abc123"

    def test_export_default_format(self) -> None:
        """Verify default format is markdown."""
        with patch.object(sys, "argv", ["deepagents", "threads", "export", "abc123"]):
            args = parse_args()
        assert args.format == "markdown"

    def test_export_json_format(self) -> None:
        """Verify -f json sets format."""
        with patch.object(
            sys, "argv", ["deepagents", "threads", "export", "abc123", "-f", "json"]
        ):
            args = parse_args()
        assert args.format == "json"

    def test_export_output_file(self) -> None:
        """Verify -o sets output path."""
        with patch.object(
            sys, "argv", ["deepagents", "threads", "export", "abc123", "-o", "out.md"]
        ):
            args = parse_args()
        assert args.output == "out.md"

    def test_export_all_options(self) -> None:
        """Verify all export options work together."""
        argv = [
            "deepagents",
            "threads",
            "export",
            "xyz789",
            "-f",
            "json",
            "-o",
            "out.json",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        assert args.thread_id == "xyz789"
        assert args.format == "json"
        assert args.output == "out.json"
