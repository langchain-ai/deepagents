"""Automatic plugin inventory inspection without installation."""

import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.widgets import OptionList, Static

from deepagents_code._env_vars import OFFLINE
from deepagents_code.app import DeepAgentsApp
from deepagents_code.plugins import discovery, store
from deepagents_code.plugins.marketplace import MarketplaceError
from deepagents_code.plugins.models import (
    MarketplacePluginEntry,
    PluginMarketplace,
    UrlPluginSource,
)
from deepagents_code.tui.modals import plugin_manager
from deepagents_code.tui.modals.plugin_manager import state
from deepagents_code.tui.modals.plugin_manager.models import (
    _ManagerState,
    _MarketplaceRow,
    _PluginRow,
)


@pytest.fixture
def row() -> _PluginRow:
    return _PluginRow(
        plugin_id="demo@official",
        description="Example plugin",
        enabled=False,
        version=None,
        author=None,
    )


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    entry = MarketplacePluginEntry(
        name="demo",
        source=UrlPluginSource(source_type="url", url="https://example.com/demo.git"),
    )
    marketplace = PluginMarketplace(
        name="official",
        root=tmp_path,
        manifest_path=tmp_path / "marketplace.json",
        metadata={},
        plugins=(entry,),
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_PLUGIN_CACHE_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(
        discovery, "_resolve_marketplace_and_entry", lambda _: (marketplace, entry)
    )
    return root


def test_remote_inspection_lists_components_without_installing(
    source: Path, row: _PluginRow, monkeypatch: pytest.MonkeyPatch
) -> None:
    (source / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"search": {"url": "https://example.com/mcp"}}})
    )
    skill = source / "skills" / "search"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: search\ndescription: Search docs\n---\nSearch documentation.\n"
    )
    hooks = source / "hooks"
    hooks.mkdir()
    (hooks / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {"hooks": [{"type": "command", "command": "exit 1"}]}
                    ]
                }
            }
        )
    )
    materialize = MagicMock(return_value=source)
    monkeypatch.setattr(state, "materialize_plugin_source", materialize)
    installed = store.load_installed_plugins()
    enabled = store.load_enabled_plugin_ids()

    preview = state._inspect_plugin(row)

    assert preview.skill_names == ("demo@official:search",)
    assert preview.mcp_server_names == ("search",)
    assert preview.hook_events == ("PreToolUse",)
    assert not preview.enabled
    assert isinstance(materialize.call_args.args[1].source, UrlPluginSource)
    assert store.load_installed_plugins() == installed
    assert store.load_enabled_plugin_ids() == enabled
    assert not (store.plugin_storage_root() / "cache").exists()
    assert not (store.plugin_storage_root() / "data").exists()


@pytest.mark.parametrize("failure", ["download", "manifest"])
def test_inspection_failures_are_not_empty_inventories(
    source: Path, row: _PluginRow, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    materialize = MagicMock(return_value=source)
    if failure == "download":
        materialize.side_effect = MarketplaceError("download failed")
        expected = "download failed"
    else:
        manifest = source / ".claude-plugin"
        manifest.mkdir()
        (manifest / "plugin.json").write_text(
            json.dumps({"name": "demo", "skills": "../outside"})
        )
        expected = "ignoring skills"
    monkeypatch.setattr(state, "materialize_plugin_source", materialize)

    with pytest.raises(MarketplaceError, match=expected):
        state._inspect_plugin(row)


@pytest.mark.parametrize("component", [".mcp.json", "hooks/hooks.json"])
@pytest.mark.parametrize("contents", ["{broken", "[]"])
def test_malformed_components_fail_inspection(
    source: Path,
    row: _PluginRow,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    contents: str,
) -> None:
    path = source / component
    path.parent.mkdir(exist_ok=True)
    path.write_text(contents)
    monkeypatch.setattr(state, "materialize_plugin_source", lambda *_a, **_kw: source)

    with pytest.raises((ValueError, MarketplaceError)):
        state._inspect_plugin(row)


@pytest.fixture
def screen(
    row: _PluginRow, monkeypatch: pytest.MonkeyPatch
) -> plugin_manager.PluginManagerScreen:
    monkeypatch.delenv(OFFLINE, raising=False)
    snapshot = _ManagerState(
        available_plugins=(row, replace(row, plugin_id="other@official")),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "example/official", 2, 0),),
        errors=(),
    )
    monkeypatch.setattr(
        plugin_manager, "_load_manager_state", MagicMock(return_value=snapshot)
    )
    monkeypatch.setattr(
        plugin_manager, "plugin_auto_update_setting", lambda: (False, "config")
    )
    return plugin_manager.PluginManagerScreen()


async def test_inspection_fetches_only_opened_plugin_and_caches_preview(
    row: _PluginRow,
    screen: plugin_manager.PluginManagerScreen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = MagicMock(
        return_value=replace(row, skill_count=0, mcp_server_names=("search",))
    )
    monkeypatch.setattr(plugin_manager, "_inspect_plugin", inspect)
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        inspect.assert_not_called()
        await pilot.press("/", "d", "e", "m", "o")
        inspect.assert_not_called()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        inspect.assert_called_once_with(row)
        assert "MCP: search" in str(
            screen.query_one("#plugin-manager-status", Static).content
        )

        screen.update_connection_state((), mcp_connecting=False)
        await asyncio.gather(*screen._refresh_tasks)
        assert "MCP: search" in str(
            screen.query_one("#plugin-manager-status", Static).content
        )
        await pilot.press("escape", "enter")
        assert "MCP: search" in str(
            screen.query_one("#plugin-manager-status", Static).content
        )
        assert inspect.call_count == 1


async def test_offline_disables_automatic_inspection_but_allows_manual_fetch(
    row: _PluginRow,
    screen: plugin_manager.PluginManagerScreen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OFFLINE, "1")
    inspect = MagicMock(return_value=replace(row, skill_count=0))
    monkeypatch.setattr(plugin_manager, "_inspect_plugin", inspect)
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("/", "d", "e", "m", "o", "enter")
        inspect.assert_not_called()
        assert "Contents not inspected" in str(
            screen.query_one("#plugin-manager-status", Static).content
        )
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        inspect.assert_called_once_with(row)


async def test_known_local_contents_do_not_fetch(
    row: _PluginRow,
    screen: plugin_manager.PluginManagerScreen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _ManagerState(
        available_plugins=(replace(row, skill_count=0, mcp_server_names=("local",)),),
        installed_plugins=(),
        marketplaces=(_MarketplaceRow("official", "./marketplace", 1, 0),),
        errors=(),
    )
    monkeypatch.setattr(
        plugin_manager, "_load_manager_state", lambda *_a, **_kw: snapshot
    )
    inspect = MagicMock()
    monkeypatch.setattr(plugin_manager, "_inspect_plugin", inspect)
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("enter")
        inspect.assert_not_called()
        assert "MCP: local" in str(
            screen.query_one("#plugin-manager-status", Static).content
        )


@pytest.mark.parametrize("fails", [False, True])
async def test_pending_inspection_allows_navigation_without_stale_updates(
    screen: plugin_manager.PluginManagerScreen,
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    requested: list[str] = []

    def inspect(selected: _PluginRow) -> _PluginRow:
        requested.append(selected.plugin_id)
        if selected.plugin_id == "other@official":
            return replace(selected, skill_count=0)
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=10):
            msg = "test did not release inspection"
            raise TimeoutError(msg)
        if fails:
            msg = "download failed"
            raise MarketplaceError(msg)
        return replace(selected, skill_count=0, mcp_server_names=("search",))

    monkeypatch.setattr(plugin_manager, "_inspect_plugin", inspect)
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        try:
            await pilot.press("/", "d", "e", "m", "o", "enter")
            await asyncio.wait_for(started.wait(), timeout=2)
            options = screen.query_one("#plugin-manager-options", OptionList)
            assert options.get_option("action:install").disabled
            assert options.option_count == 2
            assert "Downloading and inspecting contents" in str(
                screen.query_one("#plugin-manager-status", Static).content
            )
            await pilot.press("escape", "enter")
            assert requested == ["demo@official"]
            assert options.get_option("action:install").disabled
            await pilot.press(
                "escape", "/", "end", "ctrl+u", "o", "t", "h", "e", "r", "enter"
            )
            assert screen._mode == "plugin_details"
            assert screen._selected_plugin is not None
            assert screen._selected_plugin.plugin_id == "other@official"
        finally:
            release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert screen._selected_plugin.plugin_id == "other@official"
        assert screen._error is None
        assert "MCP: search" not in str(
            screen.query_one("#plugin-manager-status", Static).content
        )


async def test_malformed_mcp_type_recovers_and_allows_retry(
    source: Path,
    screen: plugin_manager.PluginManagerScreen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = source / ".mcp.json"
    config.write_text(json.dumps({"mcpServers": {"search": {"type": []}}}))
    monkeypatch.setattr(state, "materialize_plugin_source", lambda *_a, **_kw: source)
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("/", "d", "e", "m", "o", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "Could not inspect contents" in str(
            screen.query_one("#plugin-manager-error", Static).content
        )
        options = screen.query_one("#plugin-manager-options", OptionList)
        assert not options.get_option("action:inspect").disabled
        assert not options.get_option("action:install").disabled
        assert "Downloading" not in str(options.get_option("action:inspect").prompt)
        config.write_text(
            json.dumps({"mcpServers": {"search": {"url": "https://example.com/mcp"}}})
        )
        await pilot.press("up", "up", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert screen._error is None
        assert "MCP: search" in str(
            screen.query_one("#plugin-manager-status", Static).content
        )


async def test_failed_inspection_can_be_retried(
    row: _PluginRow,
    screen: plugin_manager.PluginManagerScreen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = MagicMock(
        side_effect=[MarketplaceError("download failed"), replace(row, skill_count=0)]
    )
    monkeypatch.setattr(plugin_manager, "_inspect_plugin", inspect)
    app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("/", "d", "e", "m", "o", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "download failed" in str(
            screen.query_one("#plugin-manager-error", Static).content
        )
        options = screen.query_one("#plugin-manager-options", OptionList)
        assert not options.get_option("action:inspect").disabled
        await pilot.press("up", "up", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert inspect.call_count == 2
        assert screen._error is None
        assert "No supported components" in str(
            screen.query_one("#plugin-manager-status", Static).content
        )
