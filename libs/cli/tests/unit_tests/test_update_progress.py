"""Tests for `UpdateProgressScreen`."""

from __future__ import annotations

from textual.app import App
from textual.widgets import Log, Static

from deepagents_cli.config import get_glyphs
from deepagents_cli.widgets.update_progress import UpdateProgressScreen


async def test_update_progress_screen_shows_tail_when_details_toggle(tmp_path) -> None:
    """The progress modal keeps a bounded tail hidden until details are toggled."""
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="pip install --upgrade deepagents-cli",
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
        assert "Running command: pip install --upgrade deepagents-cli" in str(
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


async def test_update_progress_screen_copies_log_path_only_in_details(tmp_path) -> None:
    """Pressing c copies the log path only when details are visible."""
    log_path = tmp_path / "update.log"
    screen = UpdateProgressScreen(
        latest="2.0.0",
        command="pip install --upgrade deepagents-cli",
        log_path=log_path,
    )

    copied: list[str] = []
    app = App()
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
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
        command="pip install --upgrade deepagents-cli",
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
