"""Tests for shared inline-prompt primitives."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import App
from textual.events import Paste

from deepagents_code.tui.widgets import _paste_textarea as paste_textarea_module
from deepagents_code.tui.widgets._inline_prompt import (
    InlinePromptCompletion,
    InlinePromptTextArea,
)

if TYPE_CHECKING:
    import pytest
    from textual.app import ComposeResult


class _PromptApp(App[None]):
    """Minimal host that records inline-prompt submissions."""

    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[str] = []

    def compose(self) -> ComposeResult:
        yield InlinePromptTextArea(id="prompt")

    def on_inline_prompt_text_area_submitted(
        self, event: InlinePromptTextArea.Submitted
    ) -> None:
        self.submissions.append(event.value)


class TestInlinePromptPaste:
    """Paste handling on the inline free-text field matches the chat input."""

    async def test_large_bracketed_paste_collapses_and_expands_on_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A big paste collapses to a placeholder that expands on submit."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            big = "line\n" * 10
            await ta._on_paste(Paste(big))
            await pilot.pause()

            assert ta.text == "[Pasted text #1 +10 lines]"
            assert ta.submitted_value == big

            await pilot.press("enter")
            await pilot.pause()

            assert app.submissions == [big]

    async def test_backspace_deletes_collapsed_placeholder_atomically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backspace removes a whole collapsed-paste token in one keypress."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await ta._on_paste(Paste("a\nb\nc\nd"))
            await pilot.pause()
            assert ta.text == "[Pasted text #1 +3 lines]"

            await pilot.press("backspace")
            await pilot.pause()

            assert ta.text == ""

    async def test_modified_backspace_deletes_collapsed_placeholder_atomically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Modified Backspace removes a whole collapsed-paste token."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )

        for key in ("ctrl+backspace", "alt+backspace"):
            app = _PromptApp()
            async with app.run_test() as pilot:
                ta = app.query_one(InlinePromptTextArea)
                ta.focus()
                await pilot.pause()

                await ta._on_paste(Paste("a\nb\nc\nd"))
                await pilot.pause()
                assert ta.text == "[Pasted text #1 +3 lines]"

                await pilot.press(key)
                await pilot.pause()

                assert ta.text == ""

    async def test_key_burst_with_newline_does_not_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-line paste replayed as key events inserts a newline, no submit."""
        # Widen the burst gap/window so wall-clock delays on slow runners still
        # register as one rapid burst.
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_ENTER_SUPPRESS_WINDOW_SECONDS", 60.0
        )
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            for char in "hello":
                await pilot.press(char)
            await pilot.press("enter")
            await pilot.press("w")
            await pilot.pause()

            assert app.submissions == []
            assert "\n" in ta.text

    async def test_deliberate_enter_submits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slow typing (no burst) keeps Enter as submit."""
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 0.0)
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            for char in "hello":
                await pilot.press(char)
            await pilot.press("enter")
            await pilot.pause()

            assert app.submissions == ["hello"]


class TestInlinePromptCompletion:
    """Resolve-at-most-once semantics, independent of call ordering."""

    async def test_resolve_delivers_to_future_set_first(self) -> None:
        """The common path: the future is wired before the result arrives."""
        completion: InlinePromptCompletion[str] = InlinePromptCompletion()
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        completion.set_future(future)

        assert completion.resolve("done") is True
        assert completion.resolved is True
        assert await future == "done"

    async def test_resolve_before_set_future_still_delivers(self) -> None:
        """A result recorded before the future is wired must not be stranded."""
        completion: InlinePromptCompletion[str] = InlinePromptCompletion()

        assert completion.resolve("done") is True
        assert completion.resolved is True

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        completion.set_future(future)

        assert await future == "done"

    async def test_second_resolve_is_ignored(self) -> None:
        """Only the first terminal result wins; later ones return `False`."""
        completion: InlinePromptCompletion[str] = InlinePromptCompletion()
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        completion.set_future(future)

        assert completion.resolve("first") is True
        assert completion.resolve("second") is False
        assert await future == "first"
