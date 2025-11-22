"""Tests for VS Code integration functionality."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents_cli.vscode_integration import (
    is_vscode_available,
    open_diff_in_vscode,
    open_file_in_vscode,
)


class TestVSCodeAvailability:
    """Test VS Code availability detection."""

    def test_vscode_available_when_code_command_exists(self) -> None:
        """Test that VS Code is detected when 'code' command exists."""
        with patch("shutil.which", return_value="/usr/bin/code"):
            assert is_vscode_available() is True

    def test_vscode_not_available_when_code_command_missing(self) -> None:
        """Test that VS Code is not detected when 'code' command is missing."""
        with patch("shutil.which", return_value=None):
            assert is_vscode_available() is False


class TestOpenDiffInVSCode:
    """Test opening diffs in VS Code."""

    def test_open_diff_creates_temp_files_and_calls_code(self, tmp_path: Path) -> None:
        """Test that diff opens successfully with proper temp files."""
        with (
            patch("shutil.which", return_value="/usr/bin/code"),
            patch("subprocess.run") as mock_run,
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp")),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = open_diff_in_vscode(
                "test.py",
                "before content",
                "after content",
            )

            assert result is True
            assert mock_run.called
            # Check that subprocess.run was called with correct arguments
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "code"
            assert call_args[1] == "--diff"
            assert "test_BEFORE.py" in call_args[2]
            assert "test_AFTER.py" in call_args[3]

    def test_open_diff_with_wait_flag(self, tmp_path: Path) -> None:
        """Test that --wait flag is added when wait=True."""
        with (
            patch("shutil.which", return_value="/usr/bin/code"),
            patch("subprocess.run") as mock_run,
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp")),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            open_diff_in_vscode(
                "test.py",
                "before",
                "after",
                wait=True,
            )

            call_args = mock_run.call_args[0][0]
            assert "--wait" in call_args

    def test_open_diff_returns_false_when_vscode_not_available(self) -> None:
        """Test that function returns False when VS Code is not available."""
        with patch("shutil.which", return_value=None):
            result = open_diff_in_vscode(
                "test.py",
                "before",
                "after",
            )
            assert result is False

    def test_open_diff_returns_false_on_subprocess_error(self, tmp_path: Path) -> None:
        """Test that function returns False when subprocess fails."""
        with (
            patch("shutil.which", return_value="/usr/bin/code"),
            patch("subprocess.run") as mock_run,
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp")),
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="Error opening VS Code")

            result = open_diff_in_vscode(
                "test.py",
                "before",
                "after",
            )

            assert result is False

    def test_open_diff_handles_path_object(self, tmp_path: Path) -> None:
        """Test that function accepts Path objects."""
        with (
            patch("shutil.which", return_value="/usr/bin/code"),
            patch("subprocess.run") as mock_run,
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp")),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = open_diff_in_vscode(
                Path("test.py"),
                "before",
                "after",
            )

            assert result is True

    def test_open_diff_preserves_file_extension(self, tmp_path: Path) -> None:
        """Test that file extension is preserved for syntax highlighting."""
        with (
            patch("shutil.which", return_value="/usr/bin/code"),
            patch("subprocess.run") as mock_run,
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "temp")),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            open_diff_in_vscode(
                "component.tsx",
                "before",
                "after",
            )

            call_args = mock_run.call_args[0][0]
            assert ".tsx" in call_args[2]
            assert ".tsx" in call_args[3]


class TestOpenFileInVSCode:
    """Test opening files in VS Code."""

    def test_open_file_calls_code_command(self) -> None:
        """Test that file opens with correct command."""
        with (
            patch("shutil.which", return_value="/usr/bin/code"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = open_file_in_vscode("test.py")

            assert result is True
            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "code"
            assert "test.py" in call_args[1]

    def test_open_file_returns_false_when_vscode_not_available(self) -> None:
        """Test that function returns False when VS Code is not available."""
        with patch("shutil.which", return_value=None):
            result = open_file_in_vscode("test.py")
            assert result is False

    def test_open_file_returns_false_on_subprocess_error(self) -> None:
        """Test that function returns False when subprocess fails."""
        with (
            patch("shutil.which", return_value="/usr/bin/code"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="Error")

            result = open_file_in_vscode("test.py")

            assert result is False

    def test_open_file_handles_path_object(self) -> None:
        """Test that function accepts Path objects."""
        with (
            patch("shutil.which", return_value="/usr/bin/code"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = open_file_in_vscode(Path("test.py"))

            assert result is True
