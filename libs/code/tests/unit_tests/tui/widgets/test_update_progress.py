"""Tests for `UpdateProgressScreen`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual.app import App
from textual.widgets import Log, Static

from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets.update_progress import UpdateProgressScreen

if TYPE_CHECKING:
    from collections.abc import Callable


async def test_update_progress_screen_shows_tail_when_details_toggle(tmp_path) -> None:
    """The progress modal keeps a bounded tail hidden until details are toggled."""
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="uv tool upgrade deepagents-code",
        log_path=tmp_path / "update.log",
        tail_limit=2,
    )

    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()
        details = screen.query(Static).filter(".up-details").first()
        log_path = screen.query(Static).filter(".up-log").first()
        assert details.display is False
        assert log_path.display is False
        assert "Running command: uv tool upgrade deepagents-code" in str(
            details.render()
        )
        assert "tail -f" not in str(details.render())

        screen.append_line("first")
        screen.append_line("second")
        screen.append_line("third")
        await pilot.press("d")
        await pilot.pause()

        tail = screen.query(Log).filter(".up-tail").first()
        assert details.display is True
        assert log_path.display is True
        assert tail.display is True
        assert tail.line_count == 2
        assert list(screen._tail) == ["second", "third"]


async def test_update_progress_screen_copies_log_path_only_in_details(
    tmp_path, monkeypatch
) -> None:
    """Pressing c copies the log path only when details are visible."""
    log_path = tmp_path / "update.log"
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="uv tool upgrade deepagents-code",
        log_path=log_path,
    )

    copied: list[str] = []
    app = App()
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert copied == []

        await pilot.press("d")
        await pilot.press("c")
        await pilot.pause()
        assert copied == [str(log_path)]


async def test_update_progress_screen_renders_markup_path_plainly(tmp_path) -> None:
    """Dynamic command and log path text must not be parsed as Rich markup."""
    log_path = tmp_path / "[/red]" / "update.log"
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="echo [/red]",
        log_path=log_path,
    )

    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()

        command = screen.query(Static).filter(".up-details").first()
        log = screen.query(Static).filter(".up-log").first()
        assert "[/red]" in str(command.render())
        assert "[/red]" in str(log.render())


async def test_update_progress_screen_close_waits_until_done(tmp_path) -> None:
    """Esc is ignored while the update is running and closes after completion."""
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="uv tool upgrade deepagents-code",
        log_path=tmp_path / "update.log",
    )

    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen

        screen.mark_success()
        spinner = screen.query(Static).filter(".up-spinner").first()
        assert str(spinner.render()) == get_glyphs().checkmark
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is app.screen_stack[0]


async def test_update_progress_screen_failure_uses_error_glyph(tmp_path) -> None:
    """Failure completion should not look visually successful."""
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="uv tool install -U deepagents-code",
        log_path=tmp_path / "update.log",
    )

    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()

        screen.mark_failure("uv tool install -U deepagents-code")
        await pilot.pause()

        status = screen.query(Static).filter(".up-status").first()
        spinner = screen.query(Static).filter(".up-spinner").first()
        assert "Update failed" in str(status.render())
        assert str(spinner.render()) == get_glyphs().error


async def test_update_progress_screen_warning_stays_visible(
    tmp_path, monkeypatch
) -> None:
    """Warning completion keeps the user action visible in the modal."""
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="uv tool upgrade deepagents-code",
        log_path=tmp_path / "update.log",
    )

    copied: list[str] = []
    app = App()
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()

        fix_command = "export PATH=/home/user/.local/bin:$PATH\nhash -r"
        screen.mark_warning(
            "Update installed, but another `dcode` is earlier on your PATH.\n"
            "Remove the shadowing binary, or put /home/user/.local/bin earlier "
            "on your PATH, then relaunch.",
            copy_text=fix_command,
        )
        await pilot.pause()

        status = screen.query(Static).filter(".up-status").first()
        details = screen.query(Static).filter(".up-details").first()
        spinner = screen.query(Static).filter(".up-spinner").first()
        help_text = screen.query(Static).filter(".up-help").first()
        assert "Update complete" not in str(status.render())
        assert "another `dcode` is earlier on your PATH" in str(status.render())
        assert "Remove the shadowing binary" in str(status.render())
        assert details.display is True
        assert str(spinner.render()) == get_glyphs().warning
        assert "c copy fix command" in str(help_text.render())

        await pilot.press("c")
        await pilot.pause()

        assert copied == [fix_command]


async def test_update_progress_screen_warning_without_copy_text_copies_log_path(
    tmp_path, monkeypatch
) -> None:
    """`mark_warning` without `copy_text` falls back to copying the log path.

    The production call site always supplies `copy_text`, but the parameter is
    optional, so this pins the documented fallback: with no fix command the
    help line reverts to "c copy log path" and `c` copies the log path rather
    than silently doing nothing.
    """
    log_path = tmp_path / "update.log"
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="uv tool install -U deepagents-code",
        log_path=log_path,
    )

    copied: list[str] = []
    app = App()
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()

        screen.mark_warning("Another `dcode` is earlier on your PATH.")
        await pilot.pause()

        spinner = screen.query(Static).filter(".up-spinner").first()
        help_text = screen.query(Static).filter(".up-help").first()
        assert str(spinner.render()) == get_glyphs().warning
        assert "c copy log path" in str(help_text.render())
        assert "c copy fix command" not in str(help_text.render())

        await pilot.press("c")
        await pilot.pause()

        assert copied == [str(log_path)]


@pytest.mark.parametrize(
    "complete",
    [
        pytest.param(lambda screen: screen.mark_success(), id="success"),
        pytest.param(
            lambda screen: screen.mark_failure("uv tool upgrade deepagents-code"),
            id="failure",
        ),
        pytest.param(lambda screen: screen.mark_warning("Heads up"), id="warning"),
    ],
)
async def test_update_progress_screen_quit_waits_until_done(
    complete: Callable[[UpdateProgressScreen], None],
    tmp_path,
) -> None:
    """Q is ignored while updating and quits the app once the modal is done.

    Every terminal state (success, failure, warning) gates quit on the same
    `_done` flag, so the shortcut must behave identically in each.
    """
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="uv tool upgrade deepagents-code",
        log_path=tmp_path / "update.log",
    )

    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()

        help_text = screen.query(Static).filter(".up-help").first()
        assert "q quit" not in str(help_text.render())

        await pilot.press("q")
        await pilot.pause()
        assert app.is_running

        complete(screen)
        await pilot.pause()
        assert "q quit" in str(help_text.render())

        await pilot.press("q")
        await pilot.pause()
        assert not app.is_running
