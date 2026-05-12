"""Unit tests for `deepagents_cli.clipboard`.

Covers the clipboard-backend fallback chain (`copy_text_to_clipboard`),
selection-driven copy with notification UX (`copy_selection_to_clipboard`),
and the OSC 52 escape envelope (`_copy_osc52`).
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from typing import Self
from unittest.mock import MagicMock, patch

from deepagents_cli.clipboard import (
    _copy_osc52,
    copy_selection_to_clipboard,
    copy_text_to_clipboard,
    logger as clipboard_logger,
)


class TestCopyTextToClipboard:
    """Test the multi-backend `copy_text_to_clipboard` fallback chain."""

    def test_returns_true_when_pyperclip_succeeds(self) -> None:
        """Stops at pyperclip when it succeeds; later backends untouched."""
        mock_app = MagicMock()

        with (
            patch("pyperclip.copy") as copy,
            patch("deepagents_cli.clipboard._copy_osc52") as osc52,
        ):
            success, error = copy_text_to_clipboard(mock_app, "hello")

        assert success is True
        assert error is None
        copy.assert_called_once_with("hello")
        mock_app.copy_to_clipboard.assert_not_called()
        osc52.assert_not_called()

    def test_falls_back_to_app_clipboard(self, caplog) -> None:
        """Uses Textual `app.copy_to_clipboard` after pyperclip raises."""
        mock_app = MagicMock()

        with (
            patch("pyperclip.copy", side_effect=RuntimeError("no pyperclip")) as copy,
            caplog.at_level(logging.DEBUG),
        ):
            success, error = copy_text_to_clipboard(mock_app, "hello")

        assert success is True
        assert error is None
        copy.assert_called_once_with("hello")
        mock_app.copy_to_clipboard.assert_called_once_with("hello")
        assert "no pyperclip" in caplog.text

    def test_falls_back_to_osc52(self, caplog) -> None:
        """Uses OSC 52 after pyperclip and app clipboard both raise."""
        mock_app = MagicMock()
        mock_app.copy_to_clipboard.side_effect = OSError("no app clipboard")

        with (
            patch("pyperclip.copy", side_effect=RuntimeError("no pyperclip")),
            patch("deepagents_cli.clipboard._copy_osc52") as osc52,
            caplog.at_level(logging.DEBUG),
        ):
            success, error = copy_text_to_clipboard(mock_app, "hello")

        assert success is True
        assert error is None
        osc52.assert_called_once_with("hello")
        assert "no pyperclip" in caplog.text
        assert "no app clipboard" in caplog.text

    def test_returns_last_error_when_all_backends_fail(self, caplog) -> None:
        """Returns `(False, last_error)` so callers can surface a reason."""
        mock_app = MagicMock()
        mock_app.copy_to_clipboard.side_effect = OSError("no app clipboard")

        with (
            patch("pyperclip.copy", side_effect=RuntimeError("no pyperclip")),
            patch(
                "deepagents_cli.clipboard._copy_osc52",
                side_effect=OSError("no tty"),
            ),
            caplog.at_level(logging.DEBUG),
        ):
            success, error = copy_text_to_clipboard(mock_app, "hello")

        assert success is False
        assert error == "no tty"
        assert "no pyperclip" in caplog.text
        assert "no app clipboard" in caplog.text
        assert "no tty" in caplog.text

    def test_returns_exception_class_name_when_message_empty(self) -> None:
        """An exception with empty `str()` still produces a non-empty reason."""
        mock_app = MagicMock()
        mock_app.copy_to_clipboard.side_effect = OSError()

        with (
            patch("pyperclip.copy", side_effect=RuntimeError()),
            patch(
                "deepagents_cli.clipboard._copy_osc52",
                side_effect=OSError(),
            ),
        ):
            success, error = copy_text_to_clipboard(mock_app, "hello")

        assert success is False
        assert error == "OSError"

    def test_pyperclip_import_error_falls_through_to_app_clipboard(self) -> None:
        """Missing `pyperclip` module skips that backend cleanly."""
        mock_app = MagicMock()

        with patch.dict(sys.modules, {"pyperclip": None}):
            success, error = copy_text_to_clipboard(mock_app, "hello")

        assert success is True
        assert error is None
        mock_app.copy_to_clipboard.assert_called_once_with("hello")

    def test_pyperclip_receives_markdown_byte_for_byte(self) -> None:
        """Markdown source reaches `pyperclip.copy` unmodified.

        Real `copy_text_to_clipboard` runs (no shim); only `pyperclip.copy`
        is patched, so this catches any future "helpful" stripping or
        normalization before the backend.
        """
        mock_app = MagicMock()
        markdown = "# Result\n\n- keep **markdown** source\n\n```py\nx = 1\n```"

        captured: list[str] = []

        with patch("pyperclip.copy", side_effect=captured.append):
            success, error = copy_text_to_clipboard(mock_app, markdown)

        assert success is True
        assert error is None
        assert captured == [markdown]
        mock_app.copy_to_clipboard.assert_not_called()


class TestCopyOsc52:
    """Direct coverage of the OSC 52 escape sequence (`_copy_osc52`)."""

    def test_emits_escape_envelope(self, monkeypatch) -> None:
        r"""Emits ``\x1b]52;c;<base64>\a`` written to ``/dev/tty``."""
        captured = io.StringIO()

        class _DummyTTY:
            def __init__(self) -> None:
                self.buffer = captured

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def write(self, s: str) -> int:
                self.buffer.write(s)
                return len(s)

            def flush(self) -> None:
                pass

        monkeypatch.delenv("TMUX", raising=False)
        text = "hello world"
        with patch("pathlib.Path.open", return_value=_DummyTTY()):
            _copy_osc52(text)

        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        assert captured.getvalue() == f"\033]52;c;{encoded}\a"

    def test_wraps_envelope_for_tmux_passthrough(self, monkeypatch) -> None:
        """Inside tmux, the OSC 52 sequence must be wrapped in DCS passthrough."""
        captured = io.StringIO()

        class _DummyTTY:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def write(self, s: str) -> int:
                captured.write(s)
                return len(s)

            def flush(self) -> None:
                pass

        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1234,0")
        text = "hi"
        with patch("pathlib.Path.open", return_value=_DummyTTY()):
            _copy_osc52(text)

        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        inner = f"\033]52;c;{encoded}\a"
        expected = f"\033Ptmux;\033{inner}\033\\"
        assert captured.getvalue() == expected


class TestCopySelectionToClipboard:
    """Selection-driven copy that delegates to `copy_text_to_clipboard`."""

    def test_delegates_and_notifies_on_success(self) -> None:
        """Delegates the side effect and emits an informational toast."""
        mock_app = MagicMock()
        selection = MagicMock(end=1)
        widget = MagicMock()
        widget.text_selection = selection
        widget.get_selection.return_value = ("selected text", None)
        mock_app.query.return_value = [widget]

        with patch(
            "deepagents_cli.clipboard.copy_text_to_clipboard",
            return_value=(True, None),
        ) as copy:
            copy_selection_to_clipboard(mock_app)

        copy.assert_called_once_with(mock_app, "selected text")
        mock_app.notify.assert_called_once()
        assert mock_app.notify.call_args.kwargs["severity"] == "information"
        assert mock_app.notify.call_args.kwargs["markup"] is False

    def test_warns_with_markup_disabled_when_helper_fails(self) -> None:
        """Failure path uses `markup=False` to stay safe under future edits."""
        mock_app = MagicMock()
        selection = MagicMock(end=1)
        widget = MagicMock()
        widget.text_selection = selection
        widget.get_selection.return_value = ("selected text", None)
        mock_app.query.return_value = [widget]

        with patch(
            "deepagents_cli.clipboard.copy_text_to_clipboard",
            return_value=(False, "no clipboard mechanism"),
        ):
            copy_selection_to_clipboard(mock_app)

        mock_app.notify.assert_called_once_with(
            "Failed to copy - no clipboard method available",
            severity="warning",
            timeout=3,
            markup=False,
        )

    def test_handles_widget_selection_failures(self, caplog) -> None:
        """A failing widget logs and is skipped, not re-raised."""
        mock_app = MagicMock()
        mock_widget = MagicMock()
        mock_widget.text_selection = MagicMock()
        mock_widget.get_selection.side_effect = AttributeError("No selection")
        mock_app.query.return_value = [mock_widget]

        with caplog.at_level(logging.DEBUG):
            copy_selection_to_clipboard(mock_app)

        assert "Failed to get selection from widget" in caplog.text
        assert "No selection" in caplog.text


class TestClipboardLogger:
    """Sanity check: module exposes a properly named logger."""

    def test_logger_exists_and_is_named(self) -> None:
        assert clipboard_logger is not None
        assert clipboard_logger.name == "deepagents_cli.clipboard"
