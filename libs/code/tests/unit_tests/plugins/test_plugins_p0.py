from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import _list_skills as list_sdk_skills
from textual.widgets import Input, OptionList

from deepagents_code.app import DeepAgentsApp
from deepagents_code.config import get_glyphs
from deepagents_code.mcp_tools import MCPServerInfo
from deepagents_code.plugins import (
    add_local_marketplace,
    add_marketplace_source,
    discover_plugins,
    install_plugin,
    list_available_plugins,
    remove_marketplace,
    set_installed_plugin_enabled,
    uninstall_plugin,
)
from deepagents_code.plugins.adapters.mcp import (
    discover_plugin_mcp_configs,
    plugin_mcp_configs,
    plugin_mcp_server_entries,
    scoped_mcp_server_name,
)
from deepagents_code.plugins.adapters.skills import (
    discover_plugin_skill_sources_and_roots,
    plugin_skill_sources,
)
from deepagents_code.plugins.adapters.skills_middleware import (
    PluginSkillsMiddleware,
    discover_skill_dirs,
)
from deepagents_code.plugins.commands_cli import execute_plugin_command
from deepagents_code.plugins.discovery import auto_update_plugins
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    load_marketplace,
    parse_marketplace_source,
)
from deepagents_code.plugins.models import (
    GithubPluginSource,
    MarketplaceRecord,
    PluginMarketplace,
)
from deepagents_code.plugins.store import (
    ensure_marketplace_cache_dir,
    get_primary_install_entry,
    load_enabled_plugin_ids,
    load_installed_plugins,
    load_marketplace_records,
    plugin_data_dir,
    sanitize_plugin_id,
    save_marketplace_record,
    set_plugin_enabled,
    versioned_cache_path,
)
from deepagents_code.tui.modals.plugin_manager import PluginManagerScreen
from deepagents_code.tui.modals.plugin_manager.models import _ManagerState
from deepagents_code.tui.modals.plugin_manager.state import (
    _list_plugin_skill_names,
    _load_manager_state,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_skill(path: Path, *, name: str = "review") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: Review code.\n---\n\nReview the code.",
        encoding="utf-8",
    )


def _make_marketplace(root: Path) -> None:
    _write_json(
        root / ".claude-plugin" / "marketplace.json",
        {
            "name": "company-tools",
            "owner": {"name": "Team"},
            "plugins": [
                {
                    "name": "quality-review-plugin",
                    "source": "./plugins/quality-review-plugin",
                    "description": "Quality review",
                }
            ],
        },
    )
    plugin = root / "plugins" / "quality-review-plugin"
    _write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": "quality-review-plugin", "version": "1.0.0"},
    )
    _write_skill(plugin / "skills" / "review" / "SKILL.md")
    _write_json(
        plugin / ".mcp.json",
        {
            "mcpServers": {
                "docs": {
                    "command": "${CLAUDE_PLUGIN_ROOT}/bin/docs",
                    "args": ["--data", "${CLAUDE_PLUGIN_DATA}"],
                    "cwd": "server",
                }
            }
        },
    )


def _install_remote_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str]:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    monkeypatch.setattr(
        "deepagents_code.plugins.discovery.materialize_marketplace_source",
        lambda _source: (load_marketplace(marketplace_root), marketplace_root),
    )
    add_marketplace_source("https://example.com/company-tools.git")
    plugin_id = "quality-review-plugin@company-tools"
    install_plugin(plugin_id)
    monkeypatch.setenv("DEEPAGENTS_CODE_PLUGIN_AUTO_UPDATE", "1")
    monkeypatch.delenv("DEEPAGENTS_CODE_OFFLINE")
    return marketplace_root, plugin_id


def _add_docs_helper_plugin(root: Path) -> None:
    """Add a second, MCP-less plugin to the `_make_marketplace` fixture."""
    marketplace_path = root / ".claude-plugin" / "marketplace.json"
    manifest = json.loads(marketplace_path.read_text(encoding="utf-8"))
    manifest["plugins"].append(
        {
            "name": "docs-helper",
            "source": "./plugins/docs-helper",
            "description": "Docs helper",
        }
    )
    _write_json(marketplace_path, manifest)
    plugin = root / "plugins" / "docs-helper"
    _write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": "docs-helper", "version": "1.0.0"},
    )
    _write_skill(plugin / "skills" / "lookup" / "SKILL.md", name="lookup")


async def test_plugin_manager_installed_selection_opens_details_not_disable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)
    install_plugin("quality-review-plugin@company-tools")

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = PluginManagerScreen(
            loaded_plugin_ids=frozenset({"quality-review-plugin@company-tools"})
        )
        app.push_screen(screen)
        await pilot.pause()

        await pilot.press("right")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        detail = str(screen.query_one("#plugin-manager-status").render())
        options = screen.query_one("#plugin-manager-options", OptionList)
        assert "quality-review-plugin @ company-tools" in detail
        assert "Installed components:" in detail
        assert "Status:" in detail
        assert "Enabled" in detail
        assert "pending /reload" not in detail
        assert "Disable plugin" in str(options.get_option_at_index(0).prompt)
        assert "quality-review-plugin@company-tools" in load_enabled_plugin_ids()


def test_invalid_unversioned_reinstall_preserves_previous_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    manifest_path = (
        marketplace_root
        / "plugins"
        / "quality-review-plugin"
        / ".claude-plugin"
        / "plugin.json"
    )
    _write_json(manifest_path, {"name": "quality-review-plugin"})
    add_local_marketplace(marketplace_root)
    plugin_id = "quality-review-plugin@company-tools"
    install_plugin(plugin_id)
    cached_skill = (
        versioned_cache_path(plugin_id, None) / "skills" / "review" / "SKILL.md"
    )
    original = cached_skill.read_text(encoding="utf-8")
    installed = load_installed_plugins()
    enabled = load_enabled_plugin_ids()
    copytree = shutil.copytree

    def copy_invalid(
        source: Path,
        destination: Path,
        *,
        symlinks: bool,
        dirs_exist_ok: bool,
    ) -> Path:
        monkeypatch.setattr(shutil, "copytree", copytree)
        try:
            copied = copytree(
                source,
                destination,
                symlinks=symlinks,
                dirs_exist_ok=dirs_exist_ok,
            )
        finally:
            monkeypatch.setattr(shutil, "copytree", copy_invalid)
        manifest = copied / ".claude-plugin" / "plugin.json"
        manifest.write_text("{", encoding="utf-8")
        return copied

    monkeypatch.setattr("deepagents_code.plugins.store.shutil.copytree", copy_invalid)
    with pytest.raises(MarketplaceError, match="Invalid JSON syntax"):
        install_plugin(plugin_id)

    assert cached_skill.read_text(encoding="utf-8") == original
    assert load_installed_plugins() == installed
    assert load_enabled_plugin_ids() == enabled


def test_install_does_not_follow_component_symlinks_outside_plugin(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    external_mcp = tmp_path / "external.mcp.json"
    _write_json(external_mcp, {"mcpServers": {"outside": {"command": "outside"}}})
    plugin_mcp = marketplace_root / "plugins" / "quality-review-plugin" / ".mcp.json"
    plugin_mcp.unlink()
    plugin_mcp.symlink_to(external_mcp)
    add_local_marketplace(marketplace_root)

    instance = install_plugin("quality-review-plugin@company-tools")

    assert (instance.root / ".mcp.json").is_symlink()
    assert instance.inventory.mcp_files == ()


def test_marketplace_list_redacts_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    save_marketplace_record(
        MarketplaceRecord(
            name="private-tools",
            source_type="url",
            source="https://user:secret@example.com/catalog?token=hidden",
            install_location="/cache/secret-derived-path",
        )
    )
    args = argparse.Namespace(
        plugin_command="marketplace",
        marketplace_command="list",
        output_format="text",
    )

    result = execute_plugin_command(args)
    manager_state = _load_manager_state()

    assert result is not None
    assert "secret" not in result
    assert "hidden" not in result
    assert "***" in result
    assert "secret" not in capsys.readouterr().out
    assert "secret" not in manager_state.marketplaces[0].source

    args.output_format = "json"
    execute_plugin_command(args)
    json_output = capsys.readouterr().out
    assert "secret" not in json_output
    assert "hidden" not in json_output
    assert "<managed cache>" in json_output


def test_marketplace_and_manifest_display_name_surface_in_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    root = tmp_path / "marketplace"
    _write_json(
        root / ".claude-plugin" / "marketplace.json",
        {
            "name": "company-tools",
            "plugins": [
                {
                    "name": "convex-plugin",
                    "displayName": "Convex",
                    "source": "./plugins/convex-plugin",
                    "description": "Backend",
                }
            ],
        },
    )
    plugin = root / "plugins" / "convex-plugin"
    _write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {"name": "convex-plugin", "displayName": "Ignored", "version": "1.0.0"},
    )
    _write_json(
        plugin / ".mcp.json",
        {"linear": {"type": "http", "url": "https://mcp.example.com"}},
    )
    add_local_marketplace(root)
    install_plugin("convex-plugin@company-tools")

    state = _load_manager_state()
    row = state.installed_plugins[0]
    assert row.display_name == "Convex"
    assert row.label == "Convex"
    assert row.mcp_server_names == ("linear",)
    assert row.mcp_login_servers == (
        scoped_mcp_server_name("convex-plugin@company-tools", "linear"),
    )
    entries = plugin_mcp_server_entries(discover_plugins().plugins[0])
    assert entries == (
        (
            "linear",
            scoped_mcp_server_name("convex-plugin@company-tools", "linear"),
            True,
        ),
    )


def test_plugin_skill_discovery_skips_symlinks_outside_source(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "plugin" / "skills"
    outside = tmp_path / "outside"
    _write_skill(outside / "review" / "SKILL.md")
    skills_root.mkdir(parents=True)
    (skills_root / "outside").symlink_to(outside, target_is_directory=True)
    backend = FilesystemBackend(root_dir=str(skills_root), virtual_mode=False)

    assert discover_skill_dirs(backend, str(skills_root)) == []


def test_plugin_skill_discovery_breaks_directory_symlink_cycles(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "plugin" / "skills"
    _write_skill(skills_root / "review" / "SKILL.md")
    (skills_root / "loop").symlink_to(skills_root, target_is_directory=True)
    backend = FilesystemBackend(root_dir=str(skills_root), virtual_mode=False)

    discovered = discover_skill_dirs(backend, str(skills_root))

    assert discovered == [(str((skills_root / "review").resolve()), ())]


def test_marketplace_credentials_are_preserved_for_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentialed = parse_marketplace_source(
        "https://user:secret@example.com/marketplace.json"
    )
    assert credentialed.value == ("https://user:secret@example.com/marketplace.json")

    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    marketplace = load_marketplace(marketplace_root)
    monkeypatch.setattr(
        "deepagents_code.plugins.discovery.materialize_marketplace_source",
        lambda _source: (marketplace, marketplace_root),
    )

    add_marketplace_source(
        "https://example.com/marketplace.json?token=secret&channel=stable"
    )

    stored = load_marketplace_records()["company-tools"].source
    assert stored == (
        "https://example.com/marketplace.json?token=secret&channel=stable"
    )

    add_marketplace_source("https://example.com/token/path-credential/marketplace.json")

    stored = load_marketplace_records()["company-tools"].source
    assert stored == "https://example.com/token/path-credential/marketplace.json"


def test_marketplace_name_cannot_contain_plugin_id_separator(tmp_path: Path) -> None:
    root = tmp_path / "marketplace"
    _write_json(
        root / ".claude-plugin" / "marketplace.json",
        {"name": "tools@team", "plugins": []},
    )

    with pytest.raises(MarketplaceError, match="Invalid plugin name"):
        load_marketplace(root)


def _make_agents_only_plugin(root: Path) -> None:
    """Marketplace plugin shaped like Claude's `pr-review-toolkit`."""
    _write_json(
        root / ".claude-plugin" / "marketplace.json",
        {
            "name": "claude-plugins-official",
            "owner": {"name": "Anthropic"},
            "plugins": [
                {
                    "name": "pr-review-toolkit",
                    "source": "./plugins/pr-review-toolkit",
                    "description": "PR review agents",
                }
            ],
        },
    )
    plugin = root / "plugins" / "pr-review-toolkit"
    _write_json(
        plugin / ".claude-plugin" / "plugin.json",
        {
            "name": "pr-review-toolkit",
            "version": "1.0.0",
            "description": "PR review agents",
        },
    )
    (plugin / "agents").mkdir(parents=True)
    (plugin / "agents" / "reviewer.md").write_text("# Reviewer\n", encoding="utf-8")
    (plugin / "commands").mkdir(parents=True)
    (plugin / "commands" / "review-pr.md").write_text("# Review\n", encoding="utf-8")


def test_inventory_reports_unsupported_agents_and_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_agents_only_plugin(marketplace_root)
    add_local_marketplace(marketplace_root)
    instance = install_plugin("pr-review-toolkit@claude-plugins-official")

    assert instance.inventory.skills == ()
    assert instance.inventory.mcp_files == ()
    assert instance.inventory.unsupported == ("agents", "commands")


def test_manager_state_surfaces_unsupported_components_for_agents_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_agents_only_plugin(marketplace_root)
    add_local_marketplace(marketplace_root)
    plugin_id = "pr-review-toolkit@claude-plugins-official"
    install_plugin(plugin_id)

    state = _load_manager_state(loaded_plugin_ids=frozenset())
    row = next(r for r in state.installed_plugins if r.plugin_id == plugin_id)

    assert row.unsupported_components == ("agents", "commands")
    assert row.skill_names == ()
    assert row.mcp_server_names == ()
    assert row.load_state == "pending_reload"


def test_manager_state_keeps_pending_reload_after_disable_while_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)
    plugin_id = "quality-review-plugin@company-tools"
    install_plugin(plugin_id)
    set_installed_plugin_enabled(plugin_id, enabled=False)

    state = _load_manager_state(loaded_plugin_ids=frozenset({plugin_id}))
    row = next(r for r in state.installed_plugins if r.plugin_id == plugin_id)

    assert row.enabled is False
    assert row.session_loaded is True
    assert row.load_state == "pending_reload"


def test_manager_state_defers_mcp_reload_hint_while_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)
    plugin_id = "quality-review-plugin@company-tools"
    install_plugin(plugin_id)
    loaded_plugin_ids = frozenset({plugin_id})

    connecting_state = _load_manager_state(
        mcp_connecting=True,
        loaded_plugin_ids=loaded_plugin_ids,
    )
    settled_state = _load_manager_state(loaded_plugin_ids=loaded_plugin_ids)
    connecting_row = next(
        row for row in connecting_state.installed_plugins if row.plugin_id == plugin_id
    )
    settled_row = next(
        row for row in settled_state.installed_plugins if row.plugin_id == plugin_id
    )

    assert connecting_row.load_state == "enabled"
    assert connecting_row.mcp_connected is None
    assert settled_row.mcp_connected is False


def test_manager_state_previews_local_discover_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)

    state = _load_manager_state()
    row = next(
        r
        for r in state.available_plugins
        if r.plugin_id == "quality-review-plugin@company-tools"
    )

    assert row.skill_count == 1
    assert any(name.endswith(":review") for name in row.skill_names)
    assert any(
        name.endswith("__docs") or name == "docs" for name in row.mcp_server_names
    )


def test_manager_preview_does_not_create_plugin_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)
    plugin_id = "quality-review-plugin@company-tools"
    data_dir = plugin_data_dir(plugin_id)

    assert not data_dir.exists()

    state = _load_manager_state()

    assert any(row.plugin_id == plugin_id for row in state.available_plugins)
    assert not data_dir.exists()


def test_manager_state_marks_installed_plugin_with_missing_cache_as_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)
    plugin_id = "quality-review-plugin@company-tools"
    install_plugin(plugin_id)

    entry = get_primary_install_entry(plugin_id)
    assert entry is not None
    shutil.rmtree(entry.install_path)

    state = _load_manager_state(loaded_plugin_ids=frozenset())
    row = next(r for r in state.installed_plugins if r.plugin_id == plugin_id)

    assert row.load_state == "error"
    assert row.load_error is not None
    assert "re-run install" in row.load_error
    # The actionable reason also reaches the Errors tab, not just the row.
    assert any("re-run install" in err for err in state.errors)
