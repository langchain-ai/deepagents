"""Tests for the plugin manager modal structure."""

from pathlib import Path

from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.content import _plugin_options
from deepagents_code.tui.modals.plugin_manager.models import _PluginRow


def test_plugin_manager_exports_root_component_with_colocated_css() -> None:
    assert PluginManagerScreen.CSS_PATH == "plugin_manager.tcss"
    assert (
        Path(PluginManagerScreen.__module__.replace(".", "/")).name == "plugin_manager"
    )


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
