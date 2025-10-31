"""Tests for CLI entry points."""

from io import StringIO
from unittest.mock import patch

from deepagents_cli.cli import cli_main


def test_cli_main_prints_message() -> None:
    """Test that cli_main prints the expected message."""
    output = StringIO()
    with patch("sys.stdout", output):
        cli_main()

    assert "I'm alive!" in output.getvalue()
