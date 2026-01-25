"""Tests for the status bar widget."""

from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_cli.widgets.status import StatusBar


class TestFormatCwd:
    """Tests for StatusBar._format_cwd method."""

    @pytest.fixture
    def status_bar(self) -> StatusBar:
        """Create a StatusBar instance for testing."""
        bar = StatusBar()
        bar._initial_cwd = "/home/user/project"
        bar.cwd = "/home/user/project"
        return bar

    @pytest.mark.parametrize(
        ("cwd_path", "home", "expected"),
        [
            ("/home/user/project/src", "/home/user", "~/project/src"),
            ("/home/user/a/b/c", "/home/user", "~/a/b/c"),
            ("/Users/john/work", "/Users/john", "~/work"),
            ("/var/log/app", "/home/user", "/var/log/app"),  # Outside home
        ],
    )
    def test_format_cwd_home_prefix(
        self, status_bar: StatusBar, cwd_path: str, home: str, expected: str
    ) -> None:
        """Test home directory prefix substitution."""
        with patch.object(Path, "home", return_value=Path(home)):
            result = status_bar._format_cwd(cwd_path)
            assert result == expected
            assert "\\" not in result

    def test_format_cwd_empty_uses_initial(self, status_bar: StatusBar) -> None:
        """Test that empty cwd_path uses initial cwd."""
        status_bar._initial_cwd = "/initial/path"
        status_bar.cwd = ""
        with patch.object(Path, "home", return_value=Path("/other")):
            result = status_bar._format_cwd("")
            assert "/initial/path" in result

    def test_format_cwd_handles_home_error(self, status_bar: StatusBar) -> None:
        """Test graceful handling when home lookup fails."""
        with patch.object(Path, "home", side_effect=RuntimeError("No home")):
            result = status_bar._format_cwd("/some/path")
            assert result == "/some/path"
