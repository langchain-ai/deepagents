"""Tests for the plugin manager modal structure."""

import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock

from textual.widgets import Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.content import _plugin_options
from deepagents_code.tui.modals.plugin_manager.models import _PluginRow


def test_plugin_manager_css_is_colocated_with_screen() -> None:
    screen_file = Path(inspect.getfile(PluginManagerScreen))
    css_path = screen_file.parent / PluginManagerScreen.CSS_PATH
    assert PluginManagerScreen.CSS_PATH == "plugin_manager.tcss"
    assert css_path.is_file(), f"expected colocated CSS at {css_path}"


def test_plugin_options_preserve_selectable_rows_and_spacers() -> None:
    rows = (
        _PluginRow("first@source", "First", False, None, None),
        _PluginRow("second@source", "Second", True, "1.0", "Author"),
    )

    options = _plugin_options(rows, action="detail", status=None)

    assert [option.id for option in options] == [
        "detail:first@source",
        "spacer:1",
        "detail:second@source",
    ]
    assert options[1].disabled


def test_plugin_manager_reports_only_relevant_changes_on_close() -> None:
    screen = PluginManagerScreen()
    screen.dismiss = MagicMock()

    screen.action_cancel()
    screen.dismiss.assert_called_once_with(False)

    screen.dismiss.reset_mock()
    screen._plugins_changed = True
    screen.action_cancel()
    screen.dismiss.assert_called_once_with(True)


async def test_plugin_manager_overlays_underlying_content() -> None:
    """Transparent modal backdrop must composite the screen underneath."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.theme = "textual-dark"
        await pilot.pause()

        await app.mount(Static("TOP_MARKER_VISIBLE", id="top-marker"))
        marker = app.query_one("#top-marker")
        marker.styles.dock = "top"
        marker.styles.height = 1
        await pilot.pause()

        app.push_screen(PluginManagerScreen())
        await pilot.pause()

        assert app.screen.styles.background.a == 0
        plain = re.sub(r"<[^>]+>", " ", app.export_screenshot())
        assert "TOP_MARKER_VISIBLE" in plain
        assert "Plugins" in plain
