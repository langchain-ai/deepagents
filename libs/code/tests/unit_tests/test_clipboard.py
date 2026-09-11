"""Unit tests for `deepagents_code.clipboard`.

Covers the clipboard-backend fallback chain (`copy_text_to_clipboard`),
selection-driven copy with notification UX (`copy_selection_to_clipboard`),
and the OSC 52 escape envelope (`_copy_osc52`).
"""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING, Self
from unittest.mock import MagicMock, patch

from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code.clipboard import _copy_osc52, copy_selection_to_clipboard

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


class TestCopyTextWithFeedback:
    """The copy-then-notify helper shared by Ctrl+C and the `[ COPY ]` button."""


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


class TestCopySelectionToClipboard:
    """Selection-driven copy that delegates to `copy_text_to_clipboard`."""

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


class TestClipboardLogger:
    """Sanity check: module exposes a properly named logger."""
