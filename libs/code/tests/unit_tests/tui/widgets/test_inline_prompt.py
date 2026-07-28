"""Tests for shared inline-prompt primitives."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from textual.app import App
from textual.events import Key, Paste

from deepagents_code import input as input_module
from deepagents_code.tui.widgets import (
    _inline_prompt as inline_prompt_module,
    _paste_textarea as paste_textarea_module,
)
from deepagents_code.tui.widgets._inline_prompt import (
    MEDIA_UNSUPPORTED_TOAST_PREFIX,
    InlinePromptCompletion,
    InlinePromptTextArea,
    _media_unsupported_toast,
)
from deepagents_code.tui.widgets._paste_textarea import PasteBurstTextArea

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.pilot import Pilot


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


async def _paste(pilot: Pilot[None], payload: str) -> None:
    """Deliver a bracketed paste the way a terminal does, through the App.

    Posting to the App is what makes these tests meaningful: `App.on_event`
    forwards `Paste` to the focused widget, and Textual's MRO dispatch then runs
    every ancestor's `_on_paste` — including `TextArea._on_paste`, which performs
    the actual insert. Calling `widget._on_paste(...)` directly skips that walk,
    so the text never lands and assertions on it pass no matter what the
    override does (notably whether it calls `prevent_default()`).

    Args:
        pilot: Pilot driving the running test app.
        payload: Raw paste payload to deliver.
    """
    pilot.app.post_message(Paste(payload))
    await pilot.pause()


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
            await _paste(pilot, big)

            assert ta.text == "[Pasted text #1 +10 lines]"
            assert ta.submitted_value == big

            await pilot.press("enter")
            await pilot.pause()

            assert app.submissions == [big]

    async def test_unquoted_key_event_paste_collapses_and_expands(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A large unquoted key-event paste reaches collapse handling."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_BURST_FLUSH_DELAY_SECONDS", 0.25
        )
        payload = "alpha\nbeta\ngamma\ndelta"
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            for char in payload:
                event = Key("enter", None) if char == "\n" else Key(char, char)
                await ta._on_key(event)

            assert ta.text == ""
            await pilot.pause(0.35)

            assert ta.text == "[Pasted text #1 +3 lines]"
            assert ta.submitted_value == payload

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

            await _paste(pilot, "a\nb\nc\nd")
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

                await _paste(pilot, "a\nb\nc\nd")
                assert ta.text == "[Pasted text #1 +3 lines]"

                await pilot.press(key)
                await pilot.pause()

                assert ta.text == ""

    async def test_modified_backspace_after_tab_deletes_placeholder_atomically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Modified Backspace preserves token integrity after a tab."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )

        for key in ("ctrl+backspace", "alt+backspace"):
            app = _PromptApp()
            async with app.run_test() as pilot:
                ta = app.query_one(InlinePromptTextArea)
                ta.focus()
                await pilot.pause()

                await _paste(pilot, "a\nb\nc\nd")
                ta.insert("\t")
                await pilot.pause()
                assert ta.text == "[Pasted text #1 +3 lines]\t"

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
            await pilot.pause(0.15)

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

    async def test_backslash_then_enter_inserts_newline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rapid backslash + enter inserts a newline instead of submitting.

        Some terminals (e.g. VSCode built-in) emit a literal backslash followed
        by enter for shift+enter; the inline prompt must collapse that pair into
        a newline like the chat input does.
        """
        # Widen the gap so wall-clock timing between pilot.press calls on slow
        # CI runners cannot trip the submit path.
        monkeypatch.setattr(paste_textarea_module, "_BACKSLASH_ENTER_GAP_SECONDS", 60.0)
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            ta.insert("hello")
            await pilot.pause()

            await pilot.press("backslash")
            await pilot.press("enter")
            await pilot.pause()

            # Exact match: the backslash is gone and exactly one newline was
            # inserted (guards against a double-fire re-adding the char or a
            # second newline).
            assert ta.text == "hello\n"
            assert app.submissions == []

    async def test_reset_clears_pending_backslash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replacing text cannot reuse a backslash timestamp from old text."""
        monkeypatch.setattr(paste_textarea_module, "_BACKSLASH_ENTER_GAP_SECONDS", 60.0)
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await pilot.press("backslash")
            ta.text = "replacement\\"
            ta.move_cursor((0, len(ta.text)))
            ta.reset_paste_state()

            await pilot.press("enter")
            await pilot.pause()

            assert app.submissions == ["replacement\\"]

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("shift+enter", id="shift_enter"),
            pytest.param("ctrl+j", id="ctrl_j"),
        ],
    )
    async def test_modifier_enter_inserts_newline(self, key: str) -> None:
        """Modifier+Enter (and Ctrl+J) insert a newline instead of submitting.

        This is the headline affordance the inline prompt inherits from
        `PasteBurstTextArea`; without it these keys would not insert a newline
        (and a bare-`enter` variant would submit). Regression guard for the
        shared `BINDINGS` / `_consume_modifier_newline` wiring.
        """
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            ta.insert("hello")
            await pilot.pause()

            await pilot.press(key)
            await pilot.pause()

            # Exact match guards against both keys double-firing (inherited
            # priority binding + explicit `_consume_modifier_newline` handler).
            assert ta.text == "hello\n"
            assert app.submissions == []

    async def test_identical_second_paste_expands_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-pasting identical content expands the placeholder in place."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            big = "line\n" * 10
            await _paste(pilot, big)
            assert ta.text == "[Pasted text #1 +10 lines]"

            # Pasting the same content again is treated as a request to see it in
            # full: the existing placeholder expands in place rather than a
            # second `#2` placeholder being added.
            await _paste(pilot, big)
            assert ta.text == big
            assert "[Pasted text #2" not in ta.text
            assert ta.submitted_value == big

    async def test_delete_key_deletes_collapsed_placeholder_atomically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forward-delete removes a whole collapsed-paste token in one keypress."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await _paste(pilot, "a\nb\nc\nd")
            assert ta.text == "[Pasted text #1 +3 lines]"

            ta.move_cursor((0, 0))
            await pilot.press("delete")
            await pilot.pause()

            assert ta.text == ""

    async def test_backspace_from_line_below_placeholder_keeps_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backspace on the line below a placeholder rejoins lines, keeps token.

        Regression: a newline right after a `[Pasted text #N]` placeholder was
        treated as an auto-inserted trailing separator, so backspacing from the
        start of the next line deleted the whole placeholder instead of only the
        line break.
        """
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await _paste(pilot, "a\nb\nc\nd")
            assert ta.text == "[Pasted text #1 +3 lines]"

            ta.insert("\n")
            await pilot.pause()
            assert ta.cursor_location == (1, 0)

            await pilot.press("backspace")
            await pilot.pause()

            assert ta.text == "[Pasted text #1 +3 lines]"
            assert ta.cursor_location == (0, len("[Pasted text #1 +3 lines]"))

    async def test_typed_placeholder_shape_is_not_atomic(self) -> None:
        """Hand-typed placeholder-shaped text deletes one char, not the token."""
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            # No paste occurred, so `_pasted_contents` is empty and this text is
            # not a bound token — backspace must edit it character by character.
            typed = "[Pasted text #1]"
            ta.text = typed
            ta.move_cursor((0, len(typed)))
            await pilot.press("backspace")
            await pilot.pause()

            assert ta.text == typed[:-1]

    async def test_small_burst_payload_inserts_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flushed burst below the collapse threshold inserts as plain text."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: True
        )
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            ta._paste_burst_buffer = "short text"
            await ta._flush_paste_burst()
            await pilot.pause()

            assert ta.text == "short text"
            assert ta.submitted_value == "short text"
            assert ta._pasted_contents == {}

    async def test_collapse_disabled_inserts_full_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With collapsing disabled, a large paste is inserted verbatim.

        The preference is cached at construction, so it is patched before the
        widget is created.
        """
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: False
        )
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            big = "line\n" * 10
            # Flush path inserts directly, so the full text is observable here
            # (the `_on_paste` fall-through defers its insert to Textual's base
            # handler via MRO dispatch).
            ta._paste_burst_buffer = big
            await ta._flush_paste_burst()
            await pilot.pause()

            assert ta.text == big
            assert "[Pasted text #1" not in ta.text
            assert ta._pasted_contents == {}
            assert ta.submitted_value == big

    async def test_bracketed_paste_not_collapsed_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With collapsing disabled, `_on_paste` defers instead of collapsing."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: False
        )
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await _paste(pilot, "line\n" * 10)

            # Only the deferral branch is asserted: no placeholder, no stored
            # content. The verbatim insert is Textual's base handler's job.
            assert "[Pasted text #1" not in ta.text
            assert ta._pasted_contents == {}


class TestInlinePromptMediaDrop:
    """Dragging a media file into a text-only prompt is rejected with a toast."""

    async def test_bracketed_paste_of_image_is_rejected(self, tmp_path: Path) -> None:
        """A dropped image path toasts and is not inserted as text."""
        img = tmp_path / "shot.png"
        img.write_bytes(b"img")
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await _paste(pilot, str(img))

            assert ta.text == ""
            latest = list(app._notifications)[-1]
            assert latest.message.startswith(MEDIA_UNSUPPORTED_TOAST_PREFIX)
            assert "shot.png" in latest.message
            assert latest.severity == "warning"

    async def test_key_burst_of_video_is_rejected(self, tmp_path: Path) -> None:
        """A dropped video path replayed as a burst toasts and is not inserted."""
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"vid")
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            # `_dispatch_burst_payload` takes a payload rather than an event, so
            # its whole contract is the early return; a direct call is faithful.
            await ta._dispatch_burst_payload(str(clip))
            await pilot.pause()

            assert ta.text == ""
            latest = list(app._notifications)[-1]
            assert latest.message.startswith(MEDIA_UNSUPPORTED_TOAST_PREFIX)
            assert "clip.mp4" in latest.message

    async def test_non_media_path_burst_is_inserted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dropped non-media file path falls through and inserts as text."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: False
        )
        doc = tmp_path / "notes.txt"
        doc.write_text("hello")
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await ta._dispatch_burst_payload(str(doc))
            await pilot.pause()

            assert ta.text == str(doc)
            assert not list(app._notifications)

    async def test_non_media_path_bracketed_paste_is_inserted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-media path delivered as a real paste still inserts as text."""
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: False
        )
        doc = tmp_path / "notes.txt"
        doc.write_text("hello")
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await _paste(pilot, str(doc))

            assert ta.text == str(doc)
            assert not list(app._notifications)

    async def test_mixed_drop_is_rejected_and_names_every_file(
        self, tmp_path: Path
    ) -> None:
        """A drop mixing media and non-media names both discarded files."""
        img = tmp_path / "shot.png"
        img.write_bytes(b"img")
        doc = tmp_path / "notes.txt"
        doc.write_text("hello")
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await _paste(pilot, f"{img} {doc}")

            # The whole payload is swallowed, so the toast has to account for the
            # non-media path too or it silently disappears.
            assert ta.text == ""
            latest = list(app._notifications)[-1]
            assert "shot.png" in latest.message
            assert "notes.txt" in latest.message

    async def test_relative_media_path_is_inserted_as_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typed relative media path is text, not a drop, so it is kept.

        Regression guard: resolving relative tokens against the working
        directory meant a user answering a free-text prompt could not paste
        `assets/logo.png` as their answer — it was swallowed by the toast.
        """
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: False
        )
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "logo.png").write_bytes(b"img")
        monkeypatch.chdir(tmp_path)
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await _paste(pilot, "assets/logo.png")

            assert ta.text == "assets/logo.png"
            assert not list(app._notifications)

    async def test_detection_failure_inserts_text_and_logs_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A parser regression falls back to text and surfaces at warning level.

        The fallback re-inserts the path this feature exists to reject, so the
        log level is the only signal that it happened — pin it, not just the
        fall-through.
        """
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: False
        )
        img = tmp_path / "shot.png"
        img.write_bytes(b"img")

        def _boom(_text: str) -> list[Path]:
            msg = "simulated parser regression"
            raise RuntimeError(msg)

        monkeypatch.setattr(input_module, "dropped_payload_paths", _boom)

        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            with caplog.at_level(logging.WARNING, logger=inline_prompt_module.__name__):
                await _paste(pilot, str(img))

            assert ta.text == str(img)
            assert not list(app._notifications)
            failures = [
                record
                for record in caplog.records
                if "Media-payload detection failed" in record.getMessage()
            ]
            assert failures, "expected a breadcrumb for the detection failure"
            assert failures[0].levelno == logging.WARNING

    async def test_quoted_image_drop_via_real_key_events_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A quoted media drop replayed as real keystrokes reaches the rejection.

        Drives the production route end to end — `_on_key` -> burst promotion ->
        flush timer -> `_dispatch_burst_payload` — rather than calling the
        dispatcher directly. A leading quote is in `PASTE_BURST_START_CHARS`
        precisely so a dropped path buffers, so this is the designed shape.
        """
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: False
        )
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_BURST_FLUSH_DELAY_SECONDS", 0.25
        )
        img = tmp_path / "shot.png"
        img.write_bytes(b"img")
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            for char in f"'{img}'":
                await ta._on_key(Key(char, char))

            assert ta.text == ""
            await pilot.pause(0.35)

            assert ta.text == ""
            latest = list(app._notifications)[-1]
            assert latest.message.startswith(MEDIA_UNSUPPORTED_TOAST_PREFIX)
            assert "shot.png" in latest.message

    async def test_pending_burst_is_flushed_before_a_media_paste_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Buffered keystrokes land even though the following paste is refused.

        Regression guard for the flush-first branch: a refusal returns early, so
        leaving a burst pending would let its timer insert the buffered text
        *after* the paste was already rejected.
        """
        monkeypatch.setattr(
            paste_textarea_module, "_collapse_pastes_enabled", lambda: False
        )
        monkeypatch.setattr(paste_textarea_module, "PASTE_BURST_CHAR_GAP_SECONDS", 60.0)
        monkeypatch.setattr(
            paste_textarea_module, "PASTE_BURST_FLUSH_DELAY_SECONDS", 30.0
        )
        img = tmp_path / "shot.png"
        img.write_bytes(b"img")
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            for char in "'abc":
                await ta._on_key(Key(char, char))
            assert ta._paste_burst_buffer, "expected a pending burst to flush"

            await _paste(pilot, str(img))

            assert ta.text == "'abc"
            assert not ta._paste_burst_buffer
            latest = list(app._notifications)[-1]
            assert latest.message.startswith(MEDIA_UNSUPPORTED_TOAST_PREFIX)

    async def test_toast_is_not_interpreted_as_markup(self, tmp_path: Path) -> None:
        """A filename containing Rich markup renders literally in the toast.

        `shot[1].png` is the standard browser-download shape and `[1]` is a valid
        Rich tag, so the toast has to opt out of markup. The flag is asserted on
        the `Notification` rather than the message text: markup is applied when
        the toast renders, so a message-only assertion passes either way.
        """
        img = tmp_path / "shot[1].png"
        img.write_bytes(b"img")
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            await _paste(pilot, str(img))

            assert ta.text == ""
            latest = list(app._notifications)[-1]
            assert "shot[1].png" in latest.message
            assert latest.markup is False


class TestMediaUnsupportedToast:
    """The toast text accounts for exactly the files it discards."""

    def test_single_file_is_singular_and_named(self) -> None:
        """One discarded file reads as singular."""
        assert _media_unsupported_toast([Path("/a/shot.png")]) == (
            f"{MEDIA_UNSUPPORTED_TOAST_PREFIX}; file not inserted: shot.png."
        )

    def test_distinct_names_are_listed_and_pluralized(self) -> None:
        """Two differently-named files read as plural and both appear."""
        assert _media_unsupported_toast(
            [Path("/a/shot.png"), Path("/a/notes.txt")]
        ) == (
            f"{MEDIA_UNSUPPORTED_TOAST_PREFIX}; "
            "files not inserted: shot.png, notes.txt."
        )

    def test_colliding_basenames_fall_back_to_full_paths(self) -> None:
        """Same-named files from different directories stay distinguishable.

        Deduping by basename alone would print one name under a plural noun,
        telling the user two files vanished while naming only one of them —
        which defeats the point of listing them at all.
        """
        assert _media_unsupported_toast([Path("/a/shot.png"), Path("/b/shot.png")]) == (
            f"{MEDIA_UNSUPPORTED_TOAST_PREFIX}; "
            "files not inserted: /a/shot.png, /b/shot.png."
        )

    def test_repeated_identical_path_is_counted_once(self) -> None:
        """The same path twice is one discarded file, not two."""
        assert _media_unsupported_toast([Path("/a/shot.png"), Path("/a/shot.png")]) == (
            f"{MEDIA_UNSUPPORTED_TOAST_PREFIX}; file not inserted: shot.png."
        )


class TestSharedBindings:
    """The newline/word-delete bindings live on the base and reach subclasses."""

    def test_base_bindings_are_the_single_source_of_truth(self) -> None:
        """`PasteBurstTextArea` owns the shared newline and word-delete keys.

        Introspecting the base protects every paste-aware subclass at once: a
        regression in these bindings would surface here before it reached the
        chat input or inline prompt.
        """
        newline_keys = {
            key.strip()
            for binding in PasteBurstTextArea.BINDINGS
            if binding.action == "insert_newline"
            for key in binding.key.split(",")
        }
        word_delete_keys = {
            key.strip()
            for binding in PasteBurstTextArea.BINDINGS
            if binding.action == "delete_word_left"
            for key in binding.key.split(",")
        }

        assert {"shift+enter", "alt+enter", "ctrl+enter", "ctrl+j"} <= newline_keys
        # `ctrl+m` is carriage return in terminals, so it must remain plain Enter.
        assert "ctrl+m" not in newline_keys
        assert newline_keys == PasteBurstTextArea._NEWLINE_KEYS
        assert {"ctrl+backspace", "alt+backspace"} <= word_delete_keys

    def test_inline_prompt_inherits_bindings(self) -> None:
        """`InlinePromptTextArea` defines no bindings of its own; it inherits them."""
        assert "BINDINGS" not in InlinePromptTextArea.__dict__
        assert InlinePromptTextArea._NEWLINE_KEYS == PasteBurstTextArea._NEWLINE_KEYS

    @pytest.mark.parametrize("key", ["ctrl+backspace", "alt+backspace"])
    async def test_modified_backspace_deletes_word_on_ordinary_text(
        self, key: str
    ) -> None:
        """The inherited word-delete binding removes the previous word.

        The other inline-prompt backspace tests only cover atomic deletion of a
        collapsed-paste placeholder; this one exercises the inherited
        `delete_word_left` binding on ordinary text.
        """
        app = _PromptApp()
        async with app.run_test() as pilot:
            ta = app.query_one(InlinePromptTextArea)
            ta.focus()
            await pilot.pause()

            ta.insert("hello world")
            await pilot.pause()

            await pilot.press(key)
            await pilot.pause()

            assert ta.text == "hello "


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
