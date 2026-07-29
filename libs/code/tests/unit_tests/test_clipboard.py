"""Unit tests for `deepagents_code.clipboard`.

Covers the clipboard-backend fallback chain (`copy_text_to_clipboard`),
selection-driven copy with notification UX (`copy_selection_to_clipboard`),
and the OSC 52 escape envelope (`_copy_osc52`).
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from typing import TYPE_CHECKING, Self
from unittest.mock import MagicMock, patch

from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code.clipboard import (
    _copy_osc52,
    copy_selection_to_clipboard,
    copy_text_to_clipboard,
    copy_text_with_feedback,
    logger as clipboard_logger,
)

if TYPE_CHECKING:
    from textual.pilot import Pilot

_BASE_TEXT = "hello base world"


class _SelectionApp(App[None]):
    """App whose base screen holds selectable text."""

    def compose(self) -> ComposeResult:
        yield Static(_BASE_TEXT, id="base-static")


class _SelectableModal(ModalScreen[None]):
    """Modal pushed over `_SelectionApp`, with its own selectable text."""

    def compose(self) -> ComposeResult:
        yield Static("modal text", id="modal-static")


async def _select_base_text(pilot: Pilot[None]) -> Static:
    """Select text on the base screen the way a user would, and return its widget.

    Returns:
        The base-screen widget left holding a live text selection.
    """
    await pilot.triple_click("#base-static")
    await pilot.pause()
    base_static = pilot.app.query_one("#base-static", Static)
    assert base_static.text_selection is not None
    return base_static


class TestCopyTextToClipboard:
    """Test the multi-backend `copy_text_to_clipboard` fallback chain."""

    def test_returns_true_when_pyperclip_succeeds(self) -> None:
        """Stops at pyperclip when it succeeds; later backends untouched."""
        mock_app = MagicMock()

        with (
            patch("pyperclip.copy") as copy,
            patch("deepagents_code.clipboard._copy_osc52") as osc52,
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
            caplog.at_level(logging.DEBUG, logger="deepagents_code"),
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
            patch("deepagents_code.clipboard._copy_osc52") as osc52,
            caplog.at_level(logging.DEBUG, logger="deepagents_code"),
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
                "deepagents_code.clipboard._copy_osc52",
                side_effect=OSError("no tty"),
            ),
            caplog.at_level(logging.DEBUG, logger="deepagents_code"),
        ):
            success, error = copy_text_to_clipboard(mock_app, "hello")

        assert success is False
        assert error == "no tty"
        assert "no pyperclip" in caplog.text
        assert "no app clipboard" in caplog.text
        assert "no tty" in caplog.text

    def test_unicode_encode_error_is_caught_not_propagated(self, caplog) -> None:
        """A backend raising `UnicodeEncodeError` is caught, not propagated.

        `_copy_osc52` does `text.encode("utf-8")`, which raises
        `UnicodeEncodeError` (a `ValueError` subclass) on lone surrogate code
        points. The loop must treat it like any other backend failure so the
        `(success, error)` contract holds instead of crashing the caller.
        """
        mock_app = MagicMock()
        mock_app.copy_to_clipboard.side_effect = OSError("no app clipboard")
        boom = UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")

        with (
            patch("pyperclip.copy", side_effect=RuntimeError("no pyperclip")),
            patch("deepagents_code.clipboard._copy_osc52", side_effect=boom),
            caplog.at_level(logging.DEBUG, logger="deepagents_code"),
        ):
            success, error = copy_text_to_clipboard(mock_app, "\ud800")

        assert success is False
        assert "surrogates not allowed" in (error or "")

    def test_returns_exception_class_name_when_message_empty(self) -> None:
        """An exception with empty `str()` still produces a non-empty reason."""
        mock_app = MagicMock()
        mock_app.copy_to_clipboard.side_effect = OSError()

        with (
            patch("pyperclip.copy", side_effect=RuntimeError()),
            patch(
                "deepagents_code.clipboard._copy_osc52",
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


class TestCopyTextWithFeedback:
    """The copy-then-notify helper shared by Ctrl+C and the `[ COPY ]` button."""

    def test_success_with_message_notifies(self) -> None:
        """A successful copy with a message emits a confirmation toast."""
        mock_app = MagicMock()

        with patch(
            "deepagents_code.clipboard.copy_text_to_clipboard",
            return_value=(True, None),
        ) as copy:
            result = copy_text_with_feedback(
                mock_app,
                "draft",
                failure_noun="input",
                success_message="Input copied to clipboard",
            )

        assert result is True
        copy.assert_called_once_with(mock_app, "draft")
        mock_app.notify.assert_called_once_with(
            "Input copied to clipboard",
            timeout=3,
            markup=False,
        )

    def test_success_without_message_is_silent(self) -> None:
        """With no success message, a successful copy emits no toast."""
        mock_app = MagicMock()

        with patch(
            "deepagents_code.clipboard.copy_text_to_clipboard",
            return_value=(True, None),
        ):
            result = copy_text_with_feedback(
                mock_app,
                "selection",
                failure_noun="selection",
            )

        assert result is True
        mock_app.notify.assert_not_called()

    def test_failure_warns_with_noun_and_markup_disabled(self) -> None:
        """A failed copy warns using the failure noun, with `markup=False`."""
        mock_app = MagicMock()

        with patch(
            "deepagents_code.clipboard.copy_text_to_clipboard",
            return_value=(False, "boom"),
        ):
            result = copy_text_with_feedback(
                mock_app,
                "draft",
                failure_noun="input",
                success_message="Input copied to clipboard",
            )

        assert result is False
        mock_app.notify.assert_called_once_with(
            "Failed to copy input: boom",
            severity="warning",
            timeout=3,
            markup=False,
        )

    def test_failure_without_error_uses_fallback_message(self) -> None:
        """A failure with no error detail falls back to the generic warning."""
        mock_app = MagicMock()

        with patch(
            "deepagents_code.clipboard.copy_text_to_clipboard",
            return_value=(False, None),
        ):
            copy_text_with_feedback(mock_app, "draft", failure_noun="input")

        mock_app.notify.assert_called_once_with(
            "Failed to copy input - no clipboard method available",
            severity="warning",
            timeout=3,
            markup=False,
        )


class TestCopyOsc52:
    """Direct coverage of the OSC 52 escape sequence (`_copy_osc52`)."""

    def test_emits_escape_envelope(self, monkeypatch) -> None:
        r"""Emits `\x1b]52;c;<base64>\a` written to `/dev/tty`."""
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
        mock_app.screen.query.return_value = [widget]
        # The scan must be screen-scoped. Without this, a regression to
        # `app.query` would fail below with the opaque "copy not called"
        # (a `MagicMock` query yields nothing) rather than naming the cause.
        mock_app.query.side_effect = AssertionError(
            "selection scan must query the screen, not the app",
        )

        with patch(
            "deepagents_code.clipboard.copy_text_to_clipboard",
            return_value=(True, None),
        ) as copy:
            copy_selection_to_clipboard(mock_app, screen=mock_app.screen)

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
        mock_app.screen.query.return_value = [widget]

        with patch(
            "deepagents_code.clipboard.copy_text_to_clipboard",
            return_value=(False, "no clipboard mechanism"),
        ):
            copy_selection_to_clipboard(mock_app, screen=mock_app.screen)

        mock_app.notify.assert_called_once_with(
            "Failed to copy - no clipboard method available",
            severity="warning",
            timeout=3,
            markup=False,
        )

    def test_copies_select_all_selection(self) -> None:
        """Textual represents triple-click/select-all as `Selection(None, None)`."""
        from textual.selection import SELECT_ALL

        mock_app = MagicMock()
        widget = MagicMock()
        widget.is_attached = True
        widget.text_selection = SELECT_ALL
        widget.get_selection.return_value = ("full widget text", None)
        mock_app.screen.query.return_value = [widget]

        with patch(
            "deepagents_code.clipboard.copy_text_to_clipboard",
            return_value=(True, None),
        ) as copy:
            copy_selection_to_clipboard(mock_app, screen=mock_app.screen)

        copy.assert_called_once_with(mock_app, "full widget text")

    def test_handles_widget_selection_failures(self, caplog) -> None:
        """A failing widget logs and is skipped, not re-raised."""
        mock_app = MagicMock()
        mock_widget = MagicMock()
        mock_widget.text_selection = MagicMock()
        mock_widget.get_selection.side_effect = AttributeError("No selection")
        mock_app.screen.query.return_value = [mock_widget]

        with caplog.at_level(logging.DEBUG, logger="deepagents_code"):
            copy_selection_to_clipboard(mock_app, screen=mock_app.screen)

        assert "Failed to get selection from widget" in caplog.text
        assert "No selection" in caplog.text

    def test_skips_detached_widget_without_reading_text_selection(self) -> None:
        """Un-attached widgets are skipped before `text_selection` is read.

        Guards the contract that `is_attached` short-circuits the property
        access — `widget.text_selection` raises `NoScreen` for detached
        widgets, so reading it would re-introduce the crash this fix
        addresses.
        """
        from unittest.mock import PropertyMock

        mock_app = MagicMock()
        detached = MagicMock()
        detached.is_attached = False
        type(detached).text_selection = PropertyMock(
            side_effect=AssertionError("text_selection must not be read"),
        )
        mock_app.screen.query.return_value = [detached]

        with patch(
            "deepagents_code.clipboard.copy_text_to_clipboard",
            return_value=(True, None),
        ) as copy:
            copy_selection_to_clipboard(mock_app, screen=mock_app.screen)

        copy.assert_not_called()

    def test_skips_widget_when_text_selection_raises_noscreen(self, caplog) -> None:
        """`NoScreen` from a lifecycle race is logged; sibling copy proceeds."""
        from unittest.mock import PropertyMock

        from textual.dom import NoScreen

        mock_app = MagicMock()

        racy = MagicMock()
        racy.is_attached = True
        type(racy).text_selection = PropertyMock(
            side_effect=NoScreen("node has no screen"),
        )

        sibling = MagicMock()
        sibling.is_attached = True
        sibling.text_selection = MagicMock(end=1)
        sibling.get_selection.return_value = ("sibling text", None)

        mock_app.screen.query.return_value = [racy, sibling]

        with (
            caplog.at_level(logging.DEBUG, logger="deepagents_code"),
            patch(
                "deepagents_code.clipboard.copy_text_to_clipboard",
                return_value=(True, None),
            ) as copy,
        ):
            copy_selection_to_clipboard(mock_app, screen=mock_app.screen)

        copy.assert_called_once_with(mock_app, "sibling text")
        assert "Skipping widget" in caplog.text


class TestSelectionCopyScreenScope:
    """The scan copies the passed screen's selection and no other screen's."""

    async def test_copies_only_the_active_screens_selection(self) -> None:
        """Scanning the modal copies its own selection, not the one below it.

        Covers both directions at once, so an over-correction that skips modal
        screens entirely fails here instead of passing a suite that only ever
        asserts the negative case.
        """
        app = _SelectionApp()
        async with app.run_test() as pilot:
            base_static = await _select_base_text(pilot)
            app.push_screen(_SelectableModal())
            await pilot.pause()
            await pilot.triple_click("#modal-static")
            await pilot.pause()

            with patch(
                "deepagents_code.clipboard.copy_text_to_clipboard",
                return_value=(True, None),
            ) as copy:
                copy_selection_to_clipboard(app, screen=app.screen)

            # The base-screen selection survives the modal, it just isn't copied.
            assert base_static.text_selection is not None
            assert copy.call_count == 1
            copied = copy.call_args.args[1]
            assert "modal text" in copied
            assert _BASE_TEXT not in copied

    async def test_skips_selection_on_screen_below_active_modal(self) -> None:
        """A selection stranded under the modal is left alone, not copied.

        The original defect: `App.query` is rooted at the app's default screen,
        so an unscoped scan reached this selection even though the click landed
        in the modal.
        """
        app = _SelectionApp()
        async with app.run_test() as pilot:
            base_static = await _select_base_text(pilot)
            app.push_screen(_SelectableModal())
            await pilot.pause()

            with patch(
                "deepagents_code.clipboard.copy_text_to_clipboard",
                return_value=(True, None),
            ) as copy:
                copy_selection_to_clipboard(app, screen=app.screen)

            assert base_static.text_selection is not None
            copy.assert_not_called()

    async def test_honors_a_screen_that_is_not_the_active_one(self) -> None:
        """The passed screen wins even when another screen is on top.

        Pins the contract that makes pinning-at-event-time work: the caller's
        screen is used verbatim, never re-resolved to whatever is active now.
        """
        app = _SelectionApp()
        async with app.run_test() as pilot:
            await _select_base_text(pilot)
            base_screen = app.screen
            app.push_screen(_SelectableModal())
            await pilot.pause()
            await pilot.triple_click("#modal-static")
            await pilot.pause()

            with patch(
                "deepagents_code.clipboard.copy_text_to_clipboard",
                return_value=(True, None),
            ) as copy:
                copy_selection_to_clipboard(app, screen=base_screen)

            assert copy.call_count == 1
            copied = copy.call_args.args[1]
            assert _BASE_TEXT in copied
            assert "modal text" not in copied


class TestAppSelectionCopy:
    """Regression coverage for click-chain selection copy timing."""

    def test_mouse_up_defers_copy_until_after_click_selection_updates(self) -> None:
        from deepagents_code.app import DeepAgentsApp

        mock_app = MagicMock()
        event = MagicMock()
        with patch("deepagents_code.clipboard.copy_selection_to_clipboard") as copy:
            DeepAgentsApp.on_mouse_up(mock_app, event)

        copy.assert_not_called()
        mock_app.call_after_refresh.assert_called_once_with(
            copy,
            mock_app,
            screen=mock_app.screen,
        )


class TestClipboardLogger:
    """Sanity check: module exposes a properly named logger."""

    def test_logger_exists_and_is_named(self) -> None:
        assert clipboard_logger is not None
        assert clipboard_logger.name == "deepagents_code.clipboard"
