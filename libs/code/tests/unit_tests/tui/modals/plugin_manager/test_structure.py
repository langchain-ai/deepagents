"""Tests for the plugin manager modal structure."""

import inspect
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Input, OptionList, Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.config import get_glyphs
from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.content import (
    _installed_plugin_details_content,
    _load_state_label,
    _marketplace_label,
    _plugin_details_content,
    _plugin_options,
    _plugin_prompt,
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
        _PluginRow(
            plugin_id="first@source",
            description="First",
            enabled=False,
            version=None,
            author=None,
            display_name="First",
        ),
        _PluginRow(
            plugin_id="second@source",
            description="Second",
            enabled=True,
            version="1.0",
            author="Author",
            display_name="Second",
        ),
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
        available_plugins=(
            _PluginRow(
                plugin_id="docs@official",
                description="Search documentation",
                enabled=False,
                version=None,
                author=None,
            ),
            _PluginRow(
                plugin_id="tests@official",
                description="Run the test suite",
                enabled=False,
                version=None,
                author=None,
            ),
        ),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "owner/official", 2, 0),),
        errors=(),
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


def test_filtered_plugins_matches_display_label() -> None:
    screen = PluginManagerScreen()
    screen._search_query = "convex"
    rows = (
        _PluginRow(
            plugin_id="cx-backend@tools",
            description="Database backend",
            enabled=False,
            version=None,
            author=None,
            display_name="Convex",
        ),
        _PluginRow(
            plugin_id="linear@tools",
            description="Issue tracker",
            enabled=False,
            version=None,
            author=None,
        ),
    )

    assert [row.plugin_id for row in screen._filtered_plugins(rows)] == [
        "cx-backend@tools"
    ]


def test_plugin_row_label_prefers_display_name() -> None:
    row = _PluginRow(
        plugin_id="convex@tools",
        description="Backend plugin",
        enabled=False,
        version=None,
        author=None,
        display_name="Convex",
    )
    assert row.label == "Convex"
    assert (
        _PluginRow(
            plugin_id="linear@tools",
            description="",
            enabled=False,
            version=None,
            author=None,
        ).label
        == "linear"
    )


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


def test_plugin_details_content_lists_preview_components() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality review",
        enabled=False,
        version="1.0.0",
        author="Team",
        skill_count=1,
        skill_names=("quality@tools:review",),
        mcp_server_names=("docs",),
    )

    content = str(_plugin_details_content(row))

    assert "Will install:" in content
    assert "Skills: quality@tools:review" in content
    assert "MCP: docs" in content
    assert "Components will be discovered at installation." not in content


def test_installed_details_explain_unsupported_components() -> None:
    row = _PluginRow(
        plugin_id="pr-review-toolkit@official",
        description="PR review",
        enabled=True,
        version="1.0.0",
        author=None,
        skill_count=0,
        unsupported_components=("agents", "commands"),
        session_loaded=True,
    )

    content = str(_installed_plugin_details_content(row))

    assert f"Status: {get_glyphs().checkmark} Enabled" in content
    assert "No supported components (skills/MCP)." in content
    assert "agents/" in content
    assert "commands/" in content
    assert "No components discovered." not in content


def test_installed_details_pending_reload_not_enabled() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=True,
        version="1.0.0",
        author=None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        session_loaded=False,
    )

    content = str(_installed_plugin_details_content(row))

    assert "Status: Installed · pending /reload" in content
    assert "Run /reload" in content
    assert f"{get_glyphs().checkmark} Enabled" not in content


def test_installed_details_pending_reload_after_disable() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=False,
        version="1.0.0",
        author=None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        session_loaded=True,
    )

    assert row.load_state == "pending_reload"
    content = str(_installed_plugin_details_content(row))

    assert "Status: Disabled · pending /reload" in content
    assert "unload this plugin" in content
    assert "Status: Disabled\n" not in content


def test_installed_details_error_state_shows_reason_and_fix() -> None:
    row = _PluginRow(
        plugin_id="broken@tools",
        description="Broken",
        enabled=True,
        version=None,
        author=None,
        session_loaded=False,
        load_error="install cache missing at /x; re-run install",
    )

    assert row.load_state == "error"
    content = str(_installed_plugin_details_content(row))

    assert "Status: Error — install cache missing at /x; re-run install" in content
    assert "Fix the error, then run /reload." in content


def test_installed_details_disabled_state_copy() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=False,
        version="1.0.0",
        author=None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        session_loaded=False,
    )

    assert row.load_state == "disabled"
    content = str(_installed_plugin_details_content(row))

    assert "Status: Disabled" in content
    assert "Enable the plugin, then run /reload to load it." in content


def test_installed_details_enabled_flags_mcp_restart() -> None:
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=True,
        version="1.0.0",
        author=None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        mcp_connected=False,
        session_loaded=True,
    )

    content = str(_installed_plugin_details_content(row))

    assert f"Status: {get_glyphs().checkmark} Enabled" in content
    assert "MCP servers need a server restart (/reload) to connect." in content


def test_load_state_label_matches_state() -> None:
    def _row(
        *, enabled: bool, session_loaded: bool, load_error: str | None = None
    ) -> _PluginRow:
        return _PluginRow(
            plugin_id="p@m",
            description="",
            enabled=enabled,
            version=None,
            author=None,
            session_loaded=session_loaded,
            load_error=load_error,
        )

    checkmark = get_glyphs().checkmark
    assert _load_state_label(_row(enabled=True, session_loaded=True)) == (
        f"{checkmark} enabled"
    )
    assert (
        _load_state_label(_row(enabled=True, session_loaded=False)) == "pending /reload"
    )
    assert _load_state_label(_row(enabled=False, session_loaded=False)) is None
    assert (
        _load_state_label(_row(enabled=True, session_loaded=True, load_error="boom"))
        == "error"
    )


def test_plugin_prompt_enabled_connection_hints() -> None:
    checkmark = get_glyphs().checkmark

    connected = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=True,
        version="1.0.0",
        author=None,
        mcp_connected=True,
        session_loaded=True,
    )
    prompt = str(_plugin_prompt(connected, status=None))
    assert f"{checkmark} enabled" in prompt
    assert f"{checkmark} connected" in prompt
    assert "restart to connect" not in prompt

    needs_restart = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=True,
        version="1.0.0",
        author=None,
        mcp_connected=False,
        session_loaded=True,
    )
    prompt = str(_plugin_prompt(needs_restart, status=None))
    assert "restart to connect" in prompt
    assert f"{checkmark} connected" not in prompt


def test_plugin_prompt_pending_reload_keeps_connected_when_still_loaded() -> None:
    """A disabled-but-still-loaded plugin keeps its live connection hint."""
    checkmark = get_glyphs().checkmark
    row = _PluginRow(
        plugin_id="quality@tools",
        description="Quality",
        enabled=False,
        version="1.0.0",
        author=None,
        mcp_connected=True,
        session_loaded=True,
    )

    assert row.load_state == "pending_reload"
    prompt = str(_plugin_prompt(row, status=None))

    assert "pending /reload" in prompt
    assert f"{checkmark} connected" in prompt


async def test_install_dismisses_reconnect_from_installed_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discover rows lack MCP metadata; install must inspect the returned instance."""
    screen = PluginManagerScreen()
    screen._selected_plugin = _PluginRow(
        plugin_id="linear@tools",
        description="Linear plugin",
        enabled=False,
        version=None,
        author=None,
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
        plugin_id="skills@tools",
        description="Skills only",
        enabled=False,
        version=None,
        author=None,
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
