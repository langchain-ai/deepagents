"""Tests for the plugin manager modal structure."""

import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock

from textual.widgets import Input, OptionList, Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.content import _plugin_options
from deepagents_code.tui.modals.plugin_manager.models import _ManagerState, _PluginRow


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


async def test_plugin_search_filters_and_clears() -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    screen = PluginManagerScreen()
    state = _ManagerState(
        (
            _PluginRow("docs@official", "Search documentation", False, None, None),
            _PluginRow("tests@official", "Run the test suite", False, None, None),
        ),
        (),
        (MagicMock(),),
        (),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        screen._state = state
        screen._refresh_view()
        search = screen.query_one("#plugin-manager-search", Input)
        options = screen.query_one("#plugin-manager-options", OptionList)

        await pilot.press("/")
        assert search.has_focus
        await pilot.press("d", "o", "c", "s")
        assert options.option_count == 1
        assert options.get_option_at_index(0).id == "detail:docs@official"

        await pilot.press("ctrl+a", "x")
        assert options.option_count == 1
        assert options.get_option_at_index(0).prompt == "No plugins match your search."

        await pilot.press("escape")
        assert search.value == ""
        assert search.has_focus
        assert options.option_count == 3
        await pilot.press("escape")
        assert options.has_focus


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
