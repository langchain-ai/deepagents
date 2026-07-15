"""Tests for the plugin manager modal structure."""

import inspect
from pathlib import Path

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
        _PluginRow("first@source", "First", False, None, None, display_name="First"),
        _PluginRow(
            "second@source", "Second", True, "1.0", "Author", display_name="Second"
        ),
    )

    options = _plugin_options(rows, action="detail", status=None)

    assert [option.id for option in options] == [
        "detail:first@source",
        "spacer:1",
        "detail:second@source",
    ]
    assert options[1].disabled


def test_plugin_row_label_prefers_display_name() -> None:
    row = _PluginRow(
        "convex@tools",
        "Backend plugin",
        False,
        None,
        None,
        display_name="Convex",
    )
    assert row.label == "Convex"
    assert _PluginRow("linear@tools", "", False, None, None).label == "linear"
