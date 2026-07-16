"""Tests for the plugin manager modal structure."""

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import OptionList

from deepagents_code.app import DeepAgentsApp
from deepagents_code.config import get_glyphs
from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.content import (
    _marketplace_label,
    _plugin_options,
)
from deepagents_code.tui.modals.plugin_manager.models import (
    _ManagerState,
    _MarketplaceRow,
    _PluginRow,
)


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


def test_marketplace_options_omit_divider_when_empty() -> None:
    screen = PluginManagerScreen()
    screen._tab = "marketplaces"
    screen._state = _ManagerState(
        available_plugins=(),
        installed_plugins=(),
        marketplaces=(),
        errors=(),
    )

    options = screen._current_options()

    assert [option.id for option in options] == ["add-marketplace"]


def test_healthy_marketplace_label_shows_available_plugins() -> None:
    row = _MarketplaceRow("healthy", "owner/healthy", 3, 0)

    label = _marketplace_label(row)

    assert not row.has_error
    assert "3 available" in label.plain
    assert get_glyphs().error not in label.plain


async def test_install_dismisses_reconnect_from_installed_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discover rows lack MCP metadata; install must inspect the returned instance."""
    screen = PluginManagerScreen()
    screen._selected_plugin = _PluginRow(
        "linear@tools",
        "Linear plugin",
        False,
        None,
        None,
        display_name="Linear",
    )
    assert screen._selected_plugin.mcp_server_names == ()

    monkeypatch.setattr(
        "deepagents_code.tui.modals.plugin_manager.install_plugin",
        lambda _plugin_id: object(),
    )
    monkeypatch.setattr(
        "deepagents_code.plugins.adapters.mcp.plugin_mcp_server_entries",
        lambda _instance: (("linear", "linear__linear@tools", True),),
    )
    monkeypatch.setattr(screen, "_refresh_state", AsyncMock())
    monkeypatch.setattr(screen, "notify", MagicMock())
    dismiss = MagicMock()
    monkeypatch.setattr(screen, "dismiss", dismiss)

    await screen._install_selected_plugin()

    dismiss.assert_called_once_with(("Linear", True))


async def test_install_keeps_manager_open_without_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = PluginManagerScreen()
    screen._selected_plugin = _PluginRow(
        "skills@tools",
        "Skills only",
        False,
        None,
        None,
        display_name="Skills",
    )

    monkeypatch.setattr(
        "deepagents_code.tui.modals.plugin_manager.install_plugin",
        lambda _plugin_id: object(),
    )
    monkeypatch.setattr(
        "deepagents_code.plugins.adapters.mcp.plugin_mcp_server_entries",
        lambda _instance: (),
    )
    monkeypatch.setattr(screen, "_refresh_state", AsyncMock())
    monkeypatch.setattr(screen, "notify", MagicMock())
    dismiss = MagicMock()
    monkeypatch.setattr(screen, "dismiss", dismiss)

    await screen._install_selected_plugin()

    dismiss.assert_not_called()


async def test_marketplace_divider_refits_on_resize() -> None:
    """The width-sized divider shrinks with the modal on a narrower terminal."""
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()

        screen._tab = "marketplaces"
        screen._state = _ManagerState(
            available_plugins=(),
            installed_plugins=(),
            marketplaces=(_MarketplaceRow("tools", "owner/repo", 2, 0),),
            errors=(),
        )
        screen._refresh_view()
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        assert options.get_option_at_index(1).id == "marketplace-divider"
        box = get_glyphs().box_horizontal
        wide_width = str(options.get_option_at_index(1).prompt).count(box)
        assert wide_width > 0

        await pilot.resize_terminal(60, 40)
        await pilot.pause()

        narrow_width = str(options.get_option_at_index(1).prompt).count(box)
        assert 0 < narrow_width < wide_width
