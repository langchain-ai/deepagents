"""Tests for the plugin manager modal structure."""

import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock

from textual.content import Content
from textual.widgets import Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.config import get_glyphs
from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.content import (
    _installed_plugin_details_content,
    _marketplace_label,
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


def _styles_for(content: Content, text: str) -> list[str]:
    start = content.plain.index(text)
    end = start + len(text)
    return [
        str(span.style)
        for span in content.spans
        if span.start <= start and span.end >= end
    ]


def test_empty_marketplace_state_offers_direct_add_action() -> None:
    screen = PluginManagerScreen()

    options = screen._current_options()

    assert [option.id for option in options] == ["empty", "add-marketplace"]
    assert options[0].disabled
    assert "No marketplaces installed" in str(options[0].prompt)
    assert str(options[1].prompt) == "+ Add marketplace"


def test_marketplace_options_include_divider_before_list() -> None:
    screen = PluginManagerScreen()
    screen._tab = "marketplaces"
    screen._state = _ManagerState(
        available_plugins=(),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("tools", "owner/repo", 2, 0),),
        errors=(),
    )

    options = screen._current_options()

    assert [option.id for option in options] == [
        "add-marketplace",
        "marketplace-divider",
        "marketplace:tools",
    ]
    assert options[1].disabled


def test_enabled_status_styles() -> None:
    row = _PluginRow("plugin@tools", "Description", True, "1.0", None)
    enabled = f"{get_glyphs().checkmark} enabled"

    assert any(
        "bold" in style for style in _styles_for(_plugin_prompt(row, status=None), enabled)
    )
    assert any(
        "$success" in style
        for style in _styles_for(
            _installed_plugin_details_content(row),
            f"{get_glyphs().checkmark} Enabled",
        )
    )


def test_marketplace_load_error_uses_error_style() -> None:
    row = _MarketplaceRow("tools", "owner/repo", None, 0, "invalid manifest")
    error_status = f"{get_glyphs().error} Error"

    assert row.has_error
    assert any(
        "$error" in style
        for style in _styles_for(_marketplace_label(row), error_status)
    )
    assert not _MarketplaceRow("healthy", "owner/healthy", 0, 0).has_error


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
