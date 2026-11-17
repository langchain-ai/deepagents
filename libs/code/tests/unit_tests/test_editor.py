"""Tests for the external editor module."""

from __future__ import annotations

import pathlib
import shutil
import sys
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code.editor import ExternalEditorError, open_in_editor


class TestResolveEditor:
    """Tests for editor resolution from environment."""


class TestEditorDisplayName:
    """Tests for safe editor names used in hints."""


class TestPrepareCommand:
    """Tests for command preparation with flag injection."""


class TestOpenInEditor:
    """Tests for the full open_in_editor flow."""

    @patch("deepagents_code.editor.subprocess.run")
    def test_handles_permission_error_on_cleanup(self, mock_run: MagicMock) -> None:
        """PermissionError during temp file cleanup should not propagate."""

        def fake_run(cmd: list[str], **_: object) -> MagicMock:
            filepath = cmd[-1]
            pathlib.Path(filepath).write_text("edited", encoding="utf-8")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run
        with (
            patch("deepagents_code.editor.resolve_editor", return_value=["nano"]),
            patch.object(
                pathlib.Path,
                "unlink",
                side_effect=PermissionError("locked"),
            ),
        ):
            result = open_in_editor("text")
        assert result == "edited"

    @pytest.mark.parametrize(
        ("found", "opens"),
        [("C:/VS Code/bin/code.CMD", True), ("./code.CMD", False)],
        ids=["launcher-on-path", "launcher-in-project"],
    )
    def test_windows_launches_editor_found_on_path(
        self, monkeypatch: pytest.MonkeyPatch, found: str, *, opens: bool
    ) -> None:
        """`EDITOR=code` opens VS Code's `code.cmd` launcher on Windows.

        Windows only starts a bare name as an `.exe`, so the stand-in for
        `subprocess.run` starts a launcher only by the path `shutil.which` found.
        A `code.cmd` in the project folder must never run in place of the editor.
        """
        launched: list[list[str]] = []

        def fake_which(name: str) -> str | None:
            return found if name == "code" else None

        def fake_run(cmd: list[str], **_: object) -> MagicMock:
            if cmd[0] != found:
                raise FileNotFoundError(2, "The system cannot find the file specified")
            launched.append(cmd)
            pathlib.Path(cmd[-1]).write_text("edited", encoding="utf-8")
            return MagicMock(returncode=0)

        monkeypatch.setenv("EDITOR", "code")
        monkeypatch.delenv("VISUAL", raising=False)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(shutil, "which", fake_which)
        monkeypatch.setattr("deepagents_code.editor.subprocess.run", fake_run)

        if opens:
            assert open_in_editor("text", raise_on_error=True) == "edited"
            assert launched[0][:2] == [found, "--wait"]
        else:
            with pytest.raises(ExternalEditorError):
                open_in_editor("text", raise_on_error=True)
            assert launched == []
