"""Tests for the external editor module."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

from deepagents_code.editor import open_in_editor


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
