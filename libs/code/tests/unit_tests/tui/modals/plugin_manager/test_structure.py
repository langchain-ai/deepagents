"""Tests for the plugin manager modal structure."""

import inspect
from pathlib import Path

from deepagents_code.config import get_glyphs
from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.content import (
    _installed_plugin_details_content,
    _load_state_label,
    _plugin_details_content,
    _plugin_options,
    _plugin_prompt,
)
from deepagents_code.tui.modals.plugin_manager.models import _PluginRow


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
        ),
        _PluginRow(
            plugin_id="second@source",
            description="Second",
            enabled=True,
            version="1.0",
            author="Author",
        ),
    )

    options = _plugin_options(rows, action="detail", status=None)

    assert [option.id for option in options] == [
        "detail:first@source",
        "spacer:1",
        "detail:second@source",
    ]
    assert options[1].disabled


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
