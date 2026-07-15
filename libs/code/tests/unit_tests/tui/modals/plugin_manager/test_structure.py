"""Tests for the plugin manager modal structure."""

import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock

from textual.widgets import Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.config import get_glyphs
from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.content import (
    _installed_plugin_details_content,
    _plugin_details_content,
    _plugin_options,
)
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
    assert "pending /reload" in str(options[2].prompt)


def test_plugin_options_show_pending_reload_when_disabled_but_loaded() -> None:
    rows = (
        _PluginRow(
            "loaded@source",
            "Still loaded",
            False,
            "1.0",
            None,
            session_loaded=True,
        ),
    )

    options = _plugin_options(rows, action="installed", status=None)

    assert "pending /reload" in str(options[0].prompt)


def test_plugin_details_content_lists_preview_components() -> None:
    row = _PluginRow(
        "quality@tools",
        "Quality review",
        False,
        "1.0.0",
        "Team",
        skill_count=1,
        skill_names=("quality@tools:review",),
        mcp_server_names=("docs",),
    )

    content = str(_plugin_details_content(row))

    assert "Will install:" in content
    assert "Skills: quality@tools:review" in content
    assert "MCP: docs" in content
    assert "Components will be discovered at installation." not in content


def test_plugin_details_content_explains_remote_or_unknown_preview() -> None:
    row = _PluginRow("remote@tools", "Remote", False, None, None)

    content = str(_plugin_details_content(row))

    assert "Will install:" in content
    assert "Skills and MCP servers if present" in content
    assert "agents/" in content
    assert "Components will be discovered at installation." not in content


def test_installed_details_explain_unsupported_components() -> None:
    row = _PluginRow(
        "pr-review-toolkit@official",
        "PR review",
        True,
        "1.0.0",
        None,
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
        "quality@tools",
        "Quality",
        True,
        "1.0.0",
        None,
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
        "quality@tools",
        "Quality",
        False,
        "1.0.0",
        None,
        skill_count=1,
        skill_names=("quality@tools:review",),
        session_loaded=True,
    )

    assert row.load_state == "pending_reload"
    content = str(_installed_plugin_details_content(row))

    assert "Status: Disabled · pending /reload" in content
    assert "unload this plugin" in content
    assert "Status: Disabled\n" not in content


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
