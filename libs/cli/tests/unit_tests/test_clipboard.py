"""Tests for clipboard functionality in deepagents-cli."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from textual.app import App
from textual.widgets import Static

from deepagents_cli.clipboard import copy_selection_to_clipboard


class TestCopySelectionToClipboard:
    """Tests for copy_selection_to_clipboard function."""

    def test_no_selection_no_copy(self) -> None:
        """Test that nothing is copied when no text is selected."""
        mock_app = MagicMock(spec=App)
        mock_app.query.return_value = []

        # Should not raise any exception
        copy_selection_to_clipboard(mock_app)

        # No copy methods should be called
        mock_app.copy_to_clipboard.assert_not_called()

    def test_copies_single_selection(self) -> None:
        """Test copying a single text selection."""
        # Mock pyperclip import to not interfere
        with patch.dict(sys.modules, {"pyperclip": None}):
            mock_app = MagicMock(spec=App)
            mock_widget = MagicMock(spec=Static)
            mock_widget.text_selection = ((0, 0), (0, 5))
            mock_widget.get_selection.return_value = ("Hello", None)

            mock_app.query.return_value = [mock_widget]
            mock_app.copy_to_clipboard.return_value = None

            copy_selection_to_clipboard(mock_app)

            mock_app.copy_to_clipboard.assert_called_once_with("Hello")
            mock_app.notify.assert_called_once()

    def test_copies_multiple_selections(self) -> None:
        """Test copying multiple text selections from different widgets."""
        with patch.dict(sys.modules, {"pyperclip": None}):
            mock_app = MagicMock(spec=App)

            mock_widget1 = MagicMock(spec=Static)
            mock_widget1.text_selection = ((0, 0), (0, 5))
            mock_widget1.get_selection.return_value = ("Hello", None)

            mock_widget2 = MagicMock(spec=Static)
            mock_widget2.text_selection = ((0, 0), (0, 5))
            mock_widget2.get_selection.return_value = ("World", None)

            mock_app.query.return_value = [mock_widget1, mock_widget2]
            mock_app.copy_to_clipboard.return_value = None

            copy_selection_to_clipboard(mock_app)

            mock_app.copy_to_clipboard.assert_called_once_with("Hello\nWorld")
            mock_app.notify.assert_called_once()

    def test_skips_widgets_without_selection(self) -> None:
        """Test that widgets without text_selection are skipped."""
        with patch.dict(sys.modules, {"pyperclip": None}):
            mock_app = MagicMock(spec=App)

            mock_widget1 = MagicMock(spec=Static)
            mock_widget1.text_selection = None

            mock_widget2 = MagicMock(spec=Static)
            mock_widget2.text_selection = ((0, 0), (0, 5))
            mock_widget2.get_selection.return_value = ("Selected", None)

            mock_app.query.return_value = [mock_widget1, mock_widget2]
            mock_app.copy_to_clipboard.return_value = None

            copy_selection_to_clipboard(mock_app)

            mock_app.copy_to_clipboard.assert_called_once_with("Selected")

    def test_handles_get_selection_error(self) -> None:
        """Test that errors from get_selection are handled gracefully."""
        with patch.dict(sys.modules, {"pyperclip": None}):
            mock_app = MagicMock(spec=App)

            mock_widget1 = MagicMock(spec=Static)
            mock_widget1.text_selection = ((0, 0), (0, 5))
            mock_widget1.get_selection.side_effect = ValueError("Test error")

            mock_widget2 = MagicMock(spec=Static)
            mock_widget2.text_selection = ((0, 0), (0, 5))
            mock_widget2.get_selection.return_value = ("Valid", None)

            mock_app.query.return_value = [mock_widget1, mock_widget2]
            mock_app.copy_to_clipboard.return_value = None

            copy_selection_to_clipboard(mock_app)

            # Should still copy from the valid widget
            mock_app.copy_to_clipboard.assert_called_once_with("Valid")

    def test_uses_pyperclip_when_available(self) -> None:
        """Test that pyperclip is preferred when available."""
        mock_app = MagicMock(spec=App)
        mock_widget = MagicMock(spec=Static)
        mock_widget.text_selection = ((0, 0), (0, 5))
        mock_widget.get_selection.return_value = ("Text", None)

        mock_app.query.return_value = [mock_widget]

        # Create a mock pyperclip module
        mock_pyperclip = MagicMock()
        mock_pyperclip.copy.return_value = None

        with patch.dict(sys.modules, {"pyperclip": mock_pyperclip}):
            copy_selection_to_clipboard(mock_app)

            # pyperclip should be tried first
            mock_pyperclip.copy.assert_called_once_with("Text")
            mock_app.notify.assert_called_once()
            # App's copy should not be called since pyperclip succeeded
            mock_app.copy_to_clipboard.assert_not_called()

    def test_fallback_to_app_clipboard(self) -> None:
        """Test fallback to app.copy_to_clipboard when pyperclip fails."""
        mock_app = MagicMock(spec=App)
        mock_widget = MagicMock(spec=Static)
        mock_widget.text_selection = ((0, 0), (0, 5))
        mock_widget.get_selection.return_value = ("Text", None)

        mock_app.query.return_value = [mock_widget]
        mock_app.copy_to_clipboard.return_value = None

        # Create a mock pyperclip that fails
        mock_pyperclip = MagicMock()
        mock_pyperclip.copy.side_effect = Exception("Pyperclip failed")

        with patch.dict(sys.modules, {"pyperclip": mock_pyperclip}):
            copy_selection_to_clipboard(mock_app)

            # Should fall back to app's clipboard
            mock_app.copy_to_clipboard.assert_called_once_with("Text")
            mock_app.notify.assert_called_once()

    def test_ignores_empty_selections(self) -> None:
        """Test that empty or whitespace-only selections are ignored."""
        mock_app = MagicMock(spec=App)

        mock_widget1 = MagicMock(spec=Static)
        mock_widget1.text_selection = ((0, 0), (0, 3))
        mock_widget1.get_selection.return_value = ("   ", None)

        mock_widget2 = MagicMock(spec=Static)
        mock_widget2.text_selection = ((0, 0), (0, 0))
        mock_widget2.get_selection.return_value = ("", None)

        mock_app.query.return_value = [mock_widget1, mock_widget2]

        copy_selection_to_clipboard(mock_app)

        # No copy should occur for whitespace-only selections
        mock_app.copy_to_clipboard.assert_not_called()
