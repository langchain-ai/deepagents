from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import shutil
from pathlib import Path
from typing import Any, cast

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import _list_skills as list_sdk_skills
from textual.widgets import Input, OptionList

from deepagents_code._env_vars import EXPERIMENTAL
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
    plugin_mcp_configs,
    scoped_mcp_server_name,
)
from deepagents_code.plugins.adapters.skills import (
    plugin_skill_sources,
    skill_name_prefix,
)
from deepagents_code.plugins.commands_cli import execute_plugin_command
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    load_marketplace,
    parse_marketplace_source,
)
from deepagents_code.plugins.models import (
    LocalMarketplaceSource,
    RepositoryMarketplaceSource,
    UrlMarketplaceSource,
)
from deepagents_code.plugins.store import (
    ensure_marketplace_cache_dir,
    get_primary_install_entry,
    load_enabled_plugin_ids,
    load_installed_plugins,
    load_marketplace_records,
    sanitize_plugin_id,
    versioned_cache_path,
)
from deepagents_code.tui.widgets.plugin_manager import (
    PluginManagerScreen,
    _ManagerState,
)


@pytest.fixture(autouse=True)
def _enable_experimental(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXPERIMENTAL, "1")


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

        screen = PluginManagerScreen()
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
        assert "Disable plugin" in str(options.get_option_at_index(0).prompt)
        assert "quality-review-plugin@company-tools" in load_enabled_plugin_ids()


async def test_plugin_manager_installed_row_shows_restart_hint_before_connect(
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

        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()

        await pilot.press("right")
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        prompt = str(options.get_option_at_index(0).prompt)
        assert "restart to connect" in prompt
        assert "connected" not in prompt.replace("restart to connect", "")


def test_local_marketplace_install_caches_and_discovers(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    config = tmp_path / "config"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state)
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_DIR", config)
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)

    marketplace = add_local_marketplace(marketplace_root)
    assert marketplace.name == "company-tools"
    assert list_available_plugins() == (
        ("quality-review-plugin@company-tools", "Quality review", False),
    )

    plugin_id = "quality-review-plugin@company-tools"
    instance = install_plugin(plugin_id)
    cache_root = versioned_cache_path(plugin_id, "1.0.0")

    assert instance.root == cache_root.resolve()
    assert cache_root.is_dir()
    assert plugin_id in load_installed_plugins()
    assert plugin_id in load_enabled_plugin_ids()

    result = discover_plugins()
    assert not result.warnings
    assert len(result.plugins) == 1
    plugin = result.plugins[0]
    assert plugin.plugin_id == plugin_id
    assert plugin.root == cache_root.resolve()
    assert plugin.inventory.skills == ((cache_root / "skills").resolve(),)
    assert plugin.inventory.mcp_files == ((cache_root / ".mcp.json").resolve(),)

    # Marketplace source tree is unchanged; agent loads from cache.
    marketplace_skills = (
        marketplace_root / "plugins" / "quality-review-plugin" / "skills"
    ).resolve()
    assert plugin.inventory.skills != (marketplace_skills,)


def test_plugin_version_comes_from_manifest(tmp_path: Path, monkeypatch) -> None:
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
    assert install_plugin(plugin_id).version == "1.0.0"
    uninstall_plugin(plugin_id)

    manifest_path = (
        marketplace_root
        / "plugins"
        / "quality-review-plugin"
        / ".claude-plugin"
        / "plugin.json"
    )
    _write_json(manifest_path, {"name": "quality-review-plugin"})
    assert install_plugin(plugin_id).version is None


def test_failed_cached_plugin_load_rolls_back_install(
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

    monkeypatch.setattr(
        "deepagents_code.plugins.discovery._plugin_from_install_path",
        lambda **_kwargs: (None, ("invalid cached plugin",)),
    )

    with pytest.raises(MarketplaceError, match="failed to load from cache"):
        install_plugin(plugin_id)

    assert plugin_id not in load_installed_plugins()
    assert plugin_id not in load_enabled_plugin_ids()
    assert not versioned_cache_path(plugin_id, "1.0.0").exists()


def test_unversioned_reinstall_refreshes_cache(tmp_path: Path, monkeypatch) -> None:
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
    source_skill = (
        marketplace_root
        / "plugins"
        / "quality-review-plugin"
        / "skills"
        / "review"
        / "SKILL.md"
    )
    source_skill.write_text(source_skill.read_text() + "\nUpdated.", encoding="utf-8")

    install_plugin(plugin_id)

    assert cached_skill.read_text(encoding="utf-8").endswith("Updated.")


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


def test_disable_keeps_install_uninstall_removes_cache(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    config = tmp_path / "config"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state)
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_DIR", config)
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)

    plugin_id = "quality-review-plugin@company-tools"
    install_plugin(plugin_id)
    cache_root = versioned_cache_path(plugin_id, "1.0.0")
    assert cache_root.is_dir()

    set_installed_plugin_enabled(plugin_id, enabled=False)
    assert plugin_id not in load_enabled_plugin_ids()
    assert plugin_id in load_installed_plugins()
    assert cache_root.is_dir()
    assert discover_plugins().plugins == ()

    uninstall_plugin(plugin_id)
    assert plugin_id not in load_installed_plugins()
    assert not cache_root.exists()
    assert get_primary_install_entry(plugin_id) is None


def test_enable_without_install_does_not_discover(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "state"
    config = tmp_path / "config"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state)
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_DIR", config)
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)

    with pytest.raises(MarketplaceError, match="is not installed"):
        set_installed_plugin_enabled(
            "quality-review-plugin@company-tools", enabled=True
        )
    result = discover_plugins()
    assert result.plugins == ()
    assert not result.warnings


def test_marketplace_source_parser_accepts_common_inputs(tmp_path: Path) -> None:
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    marketplace_file = marketplace_root / ".claude-plugin" / "marketplace.json"

    shorthand = parse_marketplace_source("example/plugins")
    assert isinstance(shorthand, RepositoryMarketplaceSource)
    assert shorthand.source_type == "github"
    ssh = parse_marketplace_source("git@github.com:example/plugins.git#main")
    assert isinstance(ssh, RepositoryMarketplaceSource)
    assert ssh.source_type == "git"
    assert ssh.ref == "main"
    github_url = parse_marketplace_source("https://github.com/owner/repo")
    assert isinstance(github_url, RepositoryMarketplaceSource)
    assert github_url.source_type == "git"
    assert github_url.value == "https://github.com/owner/repo.git"
    json_url = parse_marketplace_source("https://example.com/marketplace.json")
    assert isinstance(json_url, UrlMarketplaceSource)
    assert json_url.source_type == "url"
    directory = parse_marketplace_source(str(marketplace_root))
    marketplace_json = parse_marketplace_source(str(marketplace_file))
    assert isinstance(directory, LocalMarketplaceSource)
    assert isinstance(marketplace_json, LocalMarketplaceSource)
    assert directory.source_type == "directory"
    assert marketplace_json.source_type == "file"
    assert not hasattr(directory, "ref")
    assert not hasattr(json_url, "ref")


def test_marketplace_add_accepts_json_file_source(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "state"
    config = tmp_path / "config"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state)
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_DIR", config)
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)

    marketplace = add_marketplace_source(
        str(marketplace_root / ".claude-plugin" / "marketplace.json")
    )

    assert marketplace.name == "company-tools"
    assert list_available_plugins() == (
        ("quality-review-plugin@company-tools", "Quality review", False),
    )


def test_remove_marketplace_uninstalls_plugins_but_keeps_local_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = ensure_marketplace_cache_dir() / "team-plugins"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)
    plugin_id = "quality-review-plugin@company-tools"
    instance = install_plugin(plugin_id)

    assert remove_marketplace("company-tools") is True

    assert marketplace_root.is_dir()
    assert not instance.root.exists()
    assert "company-tools" not in load_marketplace_records()
    assert plugin_id not in load_installed_plugins()
    assert plugin_id not in load_enabled_plugin_ids()


def test_remove_marketplace_cleans_enabled_only_plugin_state(
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

    assert remove_marketplace("company-tools") is True

    assert plugin_id not in load_enabled_plugin_ids()
    assert not discover_plugins().warnings


def test_marketplace_remove_cli_reports_missing_and_removed(
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
    install_plugin("quality-review-plugin@company-tools")
    args = argparse.Namespace(
        plugin_command="marketplace",
        marketplace_command="remove",
        name="company-tools",
        output_format="text",
    )

    removed = execute_plugin_command(args)
    missing = execute_plugin_command(args)

    assert removed == ("Removed marketplace company-tools and its installed plugins.")
    assert missing == "Marketplace company-tools is not configured."


async def test_plugin_manager_confirms_marketplace_removal(
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
    install_plugin("quality-review-plugin@company-tools")

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        screen = PluginManagerScreen()
        assert len(screen._state.marketplaces) == 1
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("right")
        await pilot.pause()
        await pilot.press("right")
        await pilot.pause()
        options = screen.query_one("#plugin-manager-options", OptionList)
        assert options.option_count == 2
        options.highlighted = 1
        await pilot.press("enter")
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        assert "Remove marketplace" in str(options.get_option_at_index(0).prompt)

        await pilot.press("enter")
        await pilot.pause()
        confirmation = str(screen.query_one("#plugin-manager-status").render())
        assert "Remove marketplace company-tools?" in confirmation
        assert "uninstalls 1 plugin" in confirmation

        await pilot.press("enter")
        await pilot.pause()

    assert "company-tools" not in load_marketplace_records()
    assert not load_installed_plugins()
    assert marketplace_root.is_dir()


async def test_plugin_manager_opens_add_marketplace_input(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        assert options.option_count == 1
        assert "No plugins available" in str(options.get_option_at_index(0).prompt)

        await pilot.press("right")
        await pilot.pause()
        await pilot.press("right")
        await pilot.pause()
        assert "> Marketplaces <" in str(
            screen.query_one("#plugin-manager-tabs").render()
        )

        await pilot.press("enter")
        await pilot.pause()

        source_input = screen.query_one("#plugin-marketplace-source", Input)
        assert source_input.display is True
        assert source_input.has_focus


async def test_plugin_manager_discover_rows_show_description_below_name(
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

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        prompt = str(options.get_option_at_index(0).prompt)

        assert "quality-review-plugin · Plugin · company-tools" in prompt
        assert "\n  Quality review" in prompt


async def test_plugin_manager_tabs_label_plugins_not_discover(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()

        tabs = str(screen.query_one("#plugin-manager-tabs").render())
        assert "> Plugins <" in tabs
        assert "Discover" not in tabs


def test_plugin_mcp_config_namespaces_and_substitutes(
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
    plugin = discover_plugins().plugins[0]

    configs = plugin_mcp_configs((plugin,), project_dir=tmp_path / "project")

    servers = cast("dict[str, dict[str, Any]]", configs[0]["mcpServers"])
    server = servers[
        scoped_mcp_server_name("quality-review-plugin@company-tools", "docs")
    ]
    assert str(plugin.root) in server["command"]
    assert server["command"].endswith("/bin/docs")
    assert str(plugin.data_dir) in server["args"]
    assert server["cwd"].endswith("/server")
    assert server["env"]["CLAUDE_PLUGIN_ROOT"] == str(plugin.root)


def test_scoped_mcp_server_name_is_valid_and_collision_resistant() -> None:
    dotted = scoped_mcp_server_name("quality.review", "docs.v1")
    underscored = scoped_mcp_server_name("quality_review", "docs_v1")

    assert re.fullmatch(r"[A-Za-z0-9_-]+", dotted)
    assert dotted != underscored


def test_marketplace_source_parser_accepts_bare_relative_directory(
    tmp_path: Path, monkeypatch
) -> None:
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    monkeypatch.chdir(tmp_path)

    source = parse_marketplace_source("marketplace")
    assert source.source_type == "directory"
    assert Path(source.value) == marketplace_root.resolve()


def test_plugin_skill_namespace_qualifies_sdk_skill_name(
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
    plugin = discover_plugins().plugins[0]
    source_path, _label, namespace = plugin_skill_sources((plugin,))[0]

    skills = list_sdk_skills(
        FilesystemBackend(root_dir=source_path, virtual_mode=False), "."
    )
    assert skills[0]["name"] == "review"

    from deepagents.middleware.skills import SkillsMiddleware

    middleware = SkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False),
        sources=[(source_path, "Plugin", skill_name_prefix(namespace))],
        system_prompt=None,
    )
    update = middleware.before_agent(
        cast("Any", {"messages": []}), runtime=cast("Any", None), config={}
    )
    assert update is not None
    assert (
        update["skills_metadata"][0]["name"]
        == "quality-review-plugin@company-tools:review"
    )


def test_cache_keys_do_not_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    assert sanitize_plugin_id("foo.bar") != sanitize_plugin_id("foo-bar")
    assert versioned_cache_path("foo.bar@market", "1.0") != versioned_cache_path(
        "foo-bar@market", "1-0"
    )
    assert versioned_cache_path("foo@market", "1.0") != versioned_cache_path(
        "foo@market", "1-0"
    )


def test_namespaces_distinguish_same_named_plugins_across_marketplaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _make_marketplace(first_root)
    _make_marketplace(second_root)
    second_manifest = second_root / ".claude-plugin" / "marketplace.json"
    second_data = json.loads(second_manifest.read_text(encoding="utf-8"))
    second_data["name"] = "other-tools"
    _write_json(second_manifest, second_data)
    add_local_marketplace(first_root)
    add_local_marketplace(second_root)
    install_plugin("quality-review-plugin@company-tools")
    install_plugin("quality-review-plugin@other-tools")

    plugins = discover_plugins().plugins
    mcp_names = {
        name
        for config in plugin_mcp_configs(plugins)
        for name in cast("dict[str, object]", config["mcpServers"])
    }
    skill_namespaces = {
        namespace for _path, _label, namespace in plugin_skill_sources(plugins)
    }

    assert len(mcp_names) == 2
    assert len(skill_namespaces) == 2


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


def test_failed_marketplace_refresh_preserves_existing_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    source_root = tmp_path / "source"
    _make_marketplace(source_root)

    def clone(args: list[str]) -> None:
        shutil.copytree(source_root, Path(args[-1]), dirs_exist_ok=True)

    monkeypatch.setattr("deepagents_code.plugins.marketplace._run_git", clone)
    add_marketplace_source("owner/repo")
    install_location = Path(
        load_marketplace_records()["company-tools"].install_location
    )
    marker = install_location / "marker"
    marker.write_text("keep", encoding="utf-8")

    def fail_clone(_args: list[str]) -> None:
        msg = "clone failed"
        raise MarketplaceError(msg)

    monkeypatch.setattr("deepagents_code.plugins.marketplace._run_git", fail_clone)
    with pytest.raises(MarketplaceError, match="clone failed"):
        add_marketplace_source("owner/repo")

    assert marker.read_text(encoding="utf-8") == "keep"


def test_url_marketplace_remote_github_plugin_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    catalog = {
        "name": "remote-tools",
        "plugins": [
            {
                "name": "remote-plugin",
                "source": {"source": "github", "repo": "owner/remote-plugin"},
            }
        ],
    }
    plugin_root = tmp_path / "remote-plugin"
    _write_json(
        plugin_root / ".claude-plugin" / "plugin.json",
        {"name": "remote-plugin", "version": "1.0.0"},
    )
    _write_skill(plugin_root / "skills" / "review" / "SKILL.md")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(json.dumps(catalog).encode()),
    )

    def clone(args: list[str]) -> None:
        shutil.copytree(plugin_root, Path(args[-1]), dirs_exist_ok=True)

    monkeypatch.setattr("deepagents_code.plugins.marketplace._run_git", clone)

    add_marketplace_source("https://example.com/marketplace.json")
    instance = install_plugin("remote-plugin@remote-tools")

    assert instance.plugin_id == "remote-plugin@remote-tools"
    assert instance.inventory.skills


async def test_plugin_manager_renders_bracketed_errors_as_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepagents_code.tui.widgets.plugin_manager._load_manager_state",
        lambda *_args, **_kwargs: _ManagerState(
            available_plugins=(),
            installed_plugins=(),
            marketplaces=(),
            errors=("failure [plugin]",),
        ),
    )
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()
        screen._error = "path [plugin]"
        screen._refresh_view()
        assert "path [plugin]" in str(
            screen.query_one("#plugin-manager-error").render()
        )

        await pilot.press("right", "right")
        await pilot.pause()
        help_text = str(screen.query_one("#plugin-manager-help").render())
        assert "{glyphs.bullet}" not in help_text
        assert get_glyphs().bullet in help_text

        await pilot.press("right")
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        assert "failure [plugin]" in str(options.get_option_at_index(0).prompt)
