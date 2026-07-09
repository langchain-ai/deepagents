from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import _list_skills as list_sdk_skills
from textual.widgets import Input, OptionList

if TYPE_CHECKING:
    from deepagents_code.plugins.models import InstallScope

from deepagents_code._env_vars import EXPERIMENTAL, PLUGIN_DIRS
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import build_plugin_commands
from deepagents_code.config import get_glyphs
from deepagents_code.mcp_tools import MCPServerInfo
from deepagents_code.plugins import (
    add_local_marketplace,
    add_marketplace_source,
    disable_plugin,
    discover_plugins,
    enable_plugin,
    enable_plugin_with_scope,
    install_plugin,
    list_available_plugins,
    remove_marketplace,
    trust_plugin,
    uninstall_plugin,
)
from deepagents_code.plugins.adapters.hooks import (
    PluginHook,
    run_post_tool_hooks,
    run_pre_tool_hooks,
    run_user_prompt_hooks,
)
from deepagents_code.plugins.adapters.mcp import (
    plugin_mcp_configs,
    scoped_mcp_server_name,
)
from deepagents_code.plugins.adapters.skills import plugin_skill_sources
from deepagents_code.plugins.commands_cli import execute_plugin_command
from deepagents_code.plugins.manifest import PluginManifestError, load_manifest
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    load_marketplace,
    parse_marketplace_source,
)
from deepagents_code.plugins.runtime import (
    clear_plugin_snapshot,
    get_plugin_snapshot,
    reload_plugin_snapshot,
)
from deepagents_code.plugins.store import (
    add_installed_plugin,
    get_primary_install_entry,
    load_enabled_plugins,
    load_favorite_plugins,
    load_installed_plugins,
    load_marketplace_records,
    load_plugin_scopes,
    sanitize_plugin_id,
    set_plugin_enabled,
    versioned_cache_path,
)
from deepagents_code.plugins.substitution import substitute_string
from deepagents_code.tui.widgets.plugin_manager import (
    PluginManagerScreen,
    _ManagerState,
)


@pytest.fixture(autouse=True)
def _enable_experimental(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(EXPERIMENTAL, "1")
    monkeypatch.chdir(tmp_path)
    clear_plugin_snapshot()


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


async def test_plugin_manager_installed_tab_groups_by_scope_with_metadata(
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
    _add_docs_helper_plugin(marketplace_root)
    add_local_marketplace(marketplace_root)
    install_plugin("quality-review-plugin@company-tools", scope="project", trust=True)
    install_plugin("docs-helper@company-tools", scope="user")

    live_mcp_info = (
        MCPServerInfo(
            name=scoped_mcp_server_name("quality-review-plugin", "docs"),
            transport="stdio",
        ),
        MCPServerInfo(name="docs-langchain", transport="stdio"),
    )

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = PluginManagerScreen(mcp_server_info=live_mcp_info)
        app.push_screen(screen)
        await pilot.pause()

        await pilot.press("right")
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        prompts = [
            str(options.get_option_at_index(index).prompt)
            for index in range(options.option_count)
        ]

        assert prompts[0] == "Project"
        assert "quality-review-plugin" in prompts[1]
        assert "1 skill" in prompts[1]
        assert f"{get_glyphs().checkmark} connected" in prompts[1]
        scoped_docs = scoped_mcp_server_name("quality-review-plugin", "docs")
        assert scoped_docs in "\n".join(prompts)
        assert "docs-langchain" in "\n".join(prompts)

        assert "User" in prompts
        docs_helper_prompt = next(p for p in prompts if "docs-helper" in p)
        assert "1 skill" in docs_helper_prompt
        assert "connected" not in docs_helper_prompt

        # The first selectable row (skipping the disabled "Project" header)
        # should be highlighted, not the header itself.
        assert options.highlighted == 1


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
    install_plugin("quality-review-plugin@company-tools", trust=True)

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
        assert load_enabled_plugins().get("quality-review-plugin@company-tools")


def test_plugin_manager_lists_all_supported_components(
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
    plugin = marketplace_root / "plugins" / "quality-review-plugin"
    (plugin / "commands").mkdir()
    (plugin / "commands" / "review.md").write_text(
        "---\ndescription: Review\n---\n\nReview.", encoding="utf-8"
    )
    (plugin / "agents").mkdir()
    (plugin / "agents" / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Reviewer\n---\n\nReview.",
        encoding="utf-8",
    )
    (plugin / "hooks").mkdir()
    (plugin / "hooks" / "hooks.json").write_text("{}", encoding="utf-8")
    add_local_marketplace(marketplace_root)
    install_plugin("quality-review-plugin@company-tools", trust=True)

    from deepagents_code.tui.widgets.plugin_manager import _load_manager_state

    state = _load_manager_state()
    row = next(
        r
        for r in state.installed_plugins
        if r.plugin_id == "quality-review-plugin@company-tools"
    )
    assert row.command_count == 1
    assert row.agent_count == 1
    assert row.hook_count == 1

    detail = str(PluginManagerScreen._installed_plugin_details_content(row))
    assert "Commands: 1 command" in detail
    assert "Agents: 1 agent" in detail
    assert "Hooks: 1 hook" in detail

    prompt = str(PluginManagerScreen._plugin_prompt(row, status=None))
    assert "Plugin ·" in prompt or " · Plugin · " in prompt
    assert "1 command" in prompt
    assert "1 agent" in prompt
    assert "1 hook" in prompt


async def test_plugin_manager_installed_details_favorite_and_disable(
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
    install_plugin("quality-review-plugin@company-tools", scope="user")

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

        # Favorite (second action)
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert "quality-review-plugin@company-tools" in load_favorite_plugins()
        options = screen.query_one("#plugin-manager-options", OptionList)
        prompts = [
            str(options.get_option_at_index(index).prompt)
            for index in range(options.option_count)
        ]
        assert prompts[0] == "Favorites"
        assert "quality-review-plugin" in prompts[1]
        assert "User" not in prompts

        await pilot.press("enter")
        await pilot.pause()
        # Disable (first action)
        await pilot.press("enter")
        await pilot.pause()

        assert (
            load_enabled_plugins().get("quality-review-plugin@company-tools") is False
        )
        # Favorite pin persists across disable.
        assert "quality-review-plugin@company-tools" in load_favorite_plugins()


async def test_plugin_manager_mcp_row_opens_details(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )

    toggled: list[str] = []

    async def _toggle(name: str) -> None:
        toggled.append(name)
        await asyncio.sleep(0)

    live_mcp_info = (
        MCPServerInfo(
            name="docs-langchain",
            transport="stdio",
            tools=(),
        ),
    )

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = PluginManagerScreen(
            mcp_server_info=live_mcp_info,
            on_toggle_mcp_disable=_toggle,
        )
        app.push_screen(screen)
        await pilot.pause()

        await pilot.press("right")
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        assert "docs-langchain" in str(options.get_option_at_index(1).prompt)
        assert "MCP" in str(options.get_option_at_index(1).prompt)

        await pilot.press("enter")
        await pilot.pause()

        detail = str(screen.query_one("#plugin-manager-status").render())
        assert "docs-langchain" in detail
        assert "MCP server" in detail
        assert "Disable server" in str(
            screen.query_one("#plugin-manager-options", OptionList)
            .get_option_at_index(0)
            .prompt
        )

        await pilot.press("enter")
        await pilot.pause()
        assert toggled == ["docs-langchain"]


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
    install_plugin("quality-review-plugin@company-tools", trust=True)

    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = PluginManagerScreen()
        app.push_screen(screen)
        await pilot.pause()

        await pilot.press("right")
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        prompt = str(options.get_option_at_index(1).prompt)
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
    instance = install_plugin(plugin_id, scope="user")
    cache_root = versioned_cache_path(plugin_id, "1.0.0")

    assert instance.root == cache_root.resolve()
    assert instance.in_place is False
    assert cache_root.is_dir()
    assert plugin_id in load_installed_plugins()
    assert load_enabled_plugins().get(plugin_id) is True

    result = discover_plugins()
    assert not result.warnings
    assert len(result.plugins) == 1
    plugin = result.plugins[0]
    assert plugin.plugin_id == plugin_id
    assert plugin.root == cache_root.resolve()
    assert plugin.in_place is False
    assert plugin.inventory.skills == ((cache_root / "skills").resolve(),)
    assert plugin.inventory.mcp_files == ((cache_root / ".mcp.json").resolve(),)

    # Marketplace source tree is unchanged; agent loads from cache.
    marketplace_skills = (
        marketplace_root / "plugins" / "quality-review-plugin" / "skills"
    ).resolve()
    assert plugin.inventory.skills != (marketplace_skills,)


def test_dev_version_reinstall_refreshes_cache(tmp_path: Path, monkeypatch) -> None:
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
        versioned_cache_path(plugin_id, "dev") / "skills" / "review" / "SKILL.md"
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


def test_failed_dev_reinstall_preserves_previous_cache(
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
        versioned_cache_path(plugin_id, "dev") / "skills" / "review" / "SKILL.md"
    )
    original = cached_skill.read_text(encoding="utf-8")

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        msg = "copy failed"
        raise OSError(msg)

    monkeypatch.setattr("deepagents_code.plugins.store.shutil.copytree", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        install_plugin(plugin_id)

    assert cached_skill.read_text(encoding="utf-8") == original


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

    disable_plugin(plugin_id)
    assert load_enabled_plugins().get(plugin_id) is False
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

    enable_plugin("quality-review-plugin@company-tools")
    result = discover_plugins()
    assert result.plugins == ()
    assert any("not installed" in warning for warning in result.warnings)


def test_marketplace_source_parser_accepts_common_inputs(tmp_path: Path) -> None:
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    marketplace_file = marketplace_root / ".claude-plugin" / "marketplace.json"

    assert parse_marketplace_source("example/plugins").source_type == "github"
    ssh = parse_marketplace_source("git@github.com:example/plugins.git#main")
    assert ssh.source_type == "git"
    assert ssh.ref == "main"
    github_url = parse_marketplace_source("https://github.com/owner/repo")
    assert github_url.source_type == "git"
    assert github_url.value == "https://github.com/owner/repo.git"
    json_url = parse_marketplace_source("https://example.com/marketplace.json")
    assert json_url.source_type == "url"
    assert parse_marketplace_source(str(marketplace_root)).source_type == "directory"
    assert parse_marketplace_source(str(marketplace_file)).source_type == "file"


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


def test_strict_false_marketplace_entry_supplies_manifest(
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
    plugin_root = marketplace_root / "plugins" / "quality-review-plugin"
    (plugin_root / ".claude-plugin" / "plugin.json").unlink()
    shutil.rmtree(plugin_root / "skills")
    _write_skill(plugin_root / "extra" / "review" / "SKILL.md")
    marketplace_path = marketplace_root / ".claude-plugin" / "marketplace.json"
    data = json.loads(marketplace_path.read_text(encoding="utf-8"))
    data["plugins"][0]["strict"] = False
    data["plugins"][0]["skills"] = "./extra"
    _write_json(marketplace_path, data)
    add_local_marketplace(marketplace_root)

    instance = install_plugin("quality-review-plugin@company-tools")

    assert instance.manifest is not None
    assert instance.inventory.skills == ((instance.root / "extra").resolve(),)


def test_manifest_rejects_ambiguous_plugin_names(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugin"
    _write_json(
        plugin_root / ".claude-plugin" / "plugin.json",
        {"name": "ambiguous@plugin"},
    )

    with pytest.raises(PluginManifestError, match="Invalid plugin name"):
        load_manifest(plugin_root)


def test_substitution_includes_session_and_skill_context(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    plugin_data = tmp_path / "data"
    skill_dir = plugin_root / "skills" / "review"
    skill_dir.mkdir(parents=True)

    result = substitute_string(
        "${CLAUDE_SESSION_ID}:${CLAUDE_SKILL_DIR}:${CLAUDE_PLUGIN_DATA}",
        plugin_root=plugin_root,
        plugin_data=plugin_data,
        session_id="session-1",
        skill_dir=skill_dir,
    )

    assert result == f"session-1:{skill_dir.resolve()}:{plugin_data.resolve()}"
    assert plugin_data.is_dir()


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


async def test_plugin_manager_discover_selection_opens_scope_details(
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

        await pilot.press("enter")
        await pilot.pause()

        detail = str(screen.query_one("#plugin-manager-status").render())
        options = screen.query_one("#plugin-manager-options", OptionList)

        assert "Plugin details" in detail
        assert "quality-review-plugin" in detail
        assert "Quality review" in detail
        assert "Install for you (user scope)" in str(
            options.get_option_at_index(0).prompt
        )
        assert "project scope" in str(options.get_option_at_index(1).prompt)
        assert "local scope" in str(options.get_option_at_index(2).prompt)

        await pilot.press("enter")
        await pilot.pause()

        assert list_available_plugins() == (
            ("quality-review-plugin@company-tools", "Quality review", True),
        )
        assert "quality-review-plugin@company-tools" in load_installed_plugins()
        assert "> Installed <" in str(screen.query_one("#plugin-manager-tabs").render())
        status = str(screen.query_one("#plugin-manager-status").render())
        assert "/reload-plugins" in status


async def test_plugin_manager_details_arrow_keys_navigate_scopes(
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

        await pilot.press("enter")
        await pilot.pause()

        options = screen.query_one("#plugin-manager-options", OptionList)
        assert options.highlighted == 0

        await pilot.press("down")
        await pilot.pause()
        assert options.highlighted == 1

        await pilot.press("down")
        await pilot.pause()
        assert options.highlighted == 2

        await pilot.press("up")
        await pilot.pause()
        assert options.highlighted == 1


def test_plugin_mcp_config_scopes_and_substitutes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    add_local_marketplace(marketplace_root)
    install_plugin("quality-review-plugin@company-tools", trust=True)
    plugin = discover_plugins().plugins[0]

    configs = plugin_mcp_configs((plugin,), project_dir=tmp_path / "project")

    servers = cast("dict[str, dict[str, Any]]", configs[0]["mcpServers"])
    server = servers[scoped_mcp_server_name("quality-review-plugin", "docs")]
    assert str(plugin.root) in server["command"]
    assert server["command"].endswith("/bin/docs")
    assert str(plugin.data_dir) in server["args"]
    assert server["cwd"].endswith("/server")
    assert server["env"]["CLAUDE_PLUGIN_ROOT"] == str(plugin.root)
    assert plugin.in_place is False


def test_plugin_mcp_requires_current_surface_trust(
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

    untrusted = discover_plugins().plugins[0]
    assert untrusted.trusted is False
    assert plugin_mcp_configs((untrusted,)) == []

    trusted = trust_plugin(plugin_id)
    assert trusted.trusted is True
    assert plugin_mcp_configs((trusted,))

    (trusted.root / ".mcp.json").write_text(
        '{"mcpServers":{"changed":{"command":"changed"}}}', encoding="utf-8"
    )
    changed = discover_plugins().plugins[0]
    assert changed.trusted is False
    assert plugin_mcp_configs((changed,)) == []


def test_uninstall_refuses_to_delete_outside_plugin_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    plugin_id = "unsafe@marketplace"
    add_installed_plugin(
        plugin_id,
        scope="user",
        install_path=str(outside),
        version="1.0.0",
    )

    uninstall_plugin(plugin_id)

    assert outside.is_dir()


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


def test_project_scoped_plugin_is_inactive_outside_its_project(
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

    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    monkeypatch.chdir(project_a)
    install_plugin("quality-review-plugin@company-tools", scope="project")
    assert len(discover_plugins().plugins) == 1

    monkeypatch.chdir(project_b)
    assert discover_plugins().plugins == ()


def test_scope_enablement_uses_local_project_user_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    project = tmp_path / "project"
    project.mkdir()
    plugin_id = "plugin@marketplace"

    set_plugin_enabled(plugin_id, True, scope="user")
    set_plugin_enabled(plugin_id, False, scope="project", project_path=str(project))
    set_plugin_enabled(plugin_id, True, scope="local", project_path=str(project))

    assert load_enabled_plugins(project_path=str(project))[plugin_id] is True
    assert load_plugin_scopes(project_path=str(project))[plugin_id] == "local"

    set_plugin_enabled(plugin_id, False, scope="local", project_path=str(project))
    assert load_enabled_plugins(project_path=str(project))[plugin_id] is False


def test_session_plugin_dir_overrides_installed_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    plugin_root = tmp_path / "session-plugin"
    _write_json(
        plugin_root / ".claude-plugin" / "plugin.json",
        {"name": "session-plugin", "version": "dev"},
    )
    _write_skill(plugin_root / "skills" / "review" / "SKILL.md")
    monkeypatch.setenv(PLUGIN_DIRS, str(plugin_root))

    result = discover_plugins()

    assert len(result.plugins) == 1
    assert result.plugins[0].plugin_id == "session-plugin@inline"
    assert result.plugins[0].origin == "dev-dir"
    assert result.plugins[0].in_place is True


def test_plugin_snapshot_changes_when_session_skill_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin_root = tmp_path / "session-plugin"
    _write_json(
        plugin_root / ".claude-plugin" / "plugin.json",
        {"name": "session-plugin"},
    )
    skill_path = plugin_root / "skills" / "review" / "SKILL.md"
    _write_skill(skill_path)
    monkeypatch.setenv(PLUGIN_DIRS, str(plugin_root))

    first = reload_plugin_snapshot(project_dir=tmp_path)
    skill_path.write_text(
        skill_path.read_text(encoding="utf-8") + "\nUpdated.",
        encoding="utf-8",
    )
    second = reload_plugin_snapshot(project_dir=tmp_path)

    assert first.fingerprint != second.fingerprint


def test_failed_snapshot_reload_retains_previous_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = reload_plugin_snapshot(project_dir=tmp_path)

    def fail(*, project_dir: Path | None = None) -> None:
        del project_dir
        msg = "snapshot failed"
        raise RuntimeError(msg)

    monkeypatch.setattr("deepagents_code.plugins.runtime.build_plugin_snapshot", fail)
    with pytest.raises(RuntimeError, match="snapshot failed"):
        reload_plugin_snapshot(project_dir=tmp_path)

    assert get_plugin_snapshot(project_dir=tmp_path) is previous


def test_plugin_commands_and_agents_load_with_canonical_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    plugin_root = tmp_path / "session-plugin"
    _write_json(
        plugin_root / ".claude-plugin" / "plugin.json",
        {"name": "session-plugin"},
    )
    command_path = plugin_root / "commands" / "review.md"
    command_path.parent.mkdir(parents=True)
    command_path.write_text(
        "---\ndescription: Review code\nargument-hint: FILE\n---\n"
        "Review $ARGUMENTS in ${CLAUDE_SESSION_ID}.",
        encoding="utf-8",
    )
    agent_path = plugin_root / "agents" / "reviewer.md"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_text(
        "---\ndescription: Review code\npermissionMode: bypass\n"
        "hooks: {}\nmcpServers: {}\n---\nReview carefully.",
        encoding="utf-8",
    )
    monkeypatch.setenv(PLUGIN_DIRS, str(plugin_root))
    caplog.set_level("WARNING")

    snapshot = reload_plugin_snapshot(project_dir=tmp_path)

    assert [command.name for command in snapshot.commands] == ["session-plugin:review"]
    assert [entry.name for entry in build_plugin_commands(snapshot.commands)] == [
        "/session-plugin:review"
    ]
    assert (
        snapshot.commands[0].render("src/app.py", session_id="session-1")
        == "Review src/app.py in session-1."
    )
    assert [agent["name"] for agent in snapshot.agents] == ["session-plugin:reviewer"]
    assert snapshot.agents[0]["system_prompt"] == "Review carefully."
    assert "Ignoring permissionMode" in caplog.text
    assert "Ignoring hooks" in caplog.text
    assert "Ignoring mcpServers" in caplog.text


def test_plugin_observational_hook_receives_tool_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin_root = tmp_path / "session-plugin"
    output = tmp_path / "hook-output.json"
    script = tmp_path / "capture.py"
    script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    _write_json(
        plugin_root / ".claude-plugin" / "plugin.json",
        {"name": "session-plugin"},
    )
    _write_json(
        plugin_root / "hooks" / "hooks.json",
        {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Write",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"{sys.executable} {script} {output}",
                            }
                        ],
                    }
                ]
            }
        },
    )
    monkeypatch.setenv(PLUGIN_DIRS, str(plugin_root))
    snapshot = reload_plugin_snapshot(project_dir=tmp_path)
    assert len(snapshot.hooks) == 1

    context = run_post_tool_hooks(
        snapshot.hooks,
        tool_name="write_file",
        tool_input={"path": "a.py"},
        tool_output="updated",
        failed=False,
        session_id="session-1",
        cwd=tmp_path,
    )

    assert context == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["hook_event_name"] == "PostToolUse"
    assert payload["tool_name"] == "Write"
    assert payload["session_id"] == "session-1"


def test_plugin_pre_tool_hook_denial_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin_root = tmp_path / "session-plugin"
    script = tmp_path / "deny.py"
    script.write_text(
        "import sys\nsys.stderr.write('blocked by plugin')\nsys.exit(2)\n",
        encoding="utf-8",
    )
    _write_json(
        plugin_root / ".claude-plugin" / "plugin.json",
        {"name": "session-plugin"},
    )
    _write_json(
        plugin_root / "hooks" / "hooks.json",
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Write",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"{sys.executable} {script}",
                            }
                        ],
                    }
                ]
            }
        },
    )
    monkeypatch.setenv(PLUGIN_DIRS, str(plugin_root))
    snapshot = reload_plugin_snapshot(project_dir=tmp_path)

    result = run_pre_tool_hooks(
        snapshot.hooks,
        tool_name="write_file",
        tool_input={"path": "a.py"},
    )

    assert result.decision == "deny"
    assert result.reason == "blocked by plugin"


@pytest.mark.parametrize("decision", ["allow", "ask"])
def test_plugin_pre_tool_hook_structured_decisions(
    decision: str,
    tmp_path: Path,
) -> None:
    script = tmp_path / "decision.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'hookSpecificOutput': {"
        "'permissionDecision': sys.argv[1], "
        "'permissionDecisionReason': 'review required', "
        "'additionalContext': 'hook context'}}))\n",
        encoding="utf-8",
    )
    hook = PluginHook(
        event="tool.use",
        source_event="PreToolUse",
        command=(sys.executable, str(script), decision),
        plugin_id="plugin@inline",
        matcher="Write",
        cwd=str(tmp_path),
        blocking=True,
    )

    result = run_pre_tool_hooks(
        (hook,),
        tool_name="write_file",
        tool_input={"path": "a.py"},
        session_id="session-1",
        cwd=tmp_path,
    )

    assert result.decision == decision
    assert result.additional_context == "hook context"
    if decision == "ask":
        assert result.reason == "review required"


def test_plugin_user_prompt_hook_blocks_submission(tmp_path: Path) -> None:
    script = tmp_path / "block_prompt.py"
    script.write_text(
        "import json\n"
        "print(json.dumps({'decision': 'block', 'reason': 'prompt blocked'}))\n",
        encoding="utf-8",
    )
    hook = PluginHook(
        event="user.prompt",
        source_event="UserPromptSubmit",
        command=(sys.executable, str(script)),
        plugin_id="plugin@inline",
        cwd=str(tmp_path),
    )

    result = run_user_prompt_hooks(
        (hook,),
        prompt="dangerous prompt",
        session_id="session-1",
        cwd=tmp_path,
    )

    assert result.decision == "deny"
    assert result.reason == "prompt blocked"


def test_remove_marketplace_removes_installs_and_cache(
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
    instance = install_plugin(plugin_id)

    assert remove_marketplace("company-tools") is True
    assert "company-tools" not in load_marketplace_records()
    assert plugin_id not in load_installed_plugins()
    assert not instance.root.exists()


def test_plugin_skill_source_prefixes_sdk_skill_name(
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
    source_path, _label, prefix = plugin_skill_sources((plugin,))[0]

    skills = list_sdk_skills(
        FilesystemBackend(root_dir=source_path, virtual_mode=False), "."
    )
    assert skills[0]["name"] == "review"

    from deepagents.middleware.skills import SkillsMiddleware

    middleware = SkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False),
        sources=[(source_path, "Plugin", prefix)],
        system_prompt=None,
    )
    update = middleware.before_agent(
        cast("Any", {"messages": []}), runtime=cast("Any", None), config={}
    )
    assert update is not None
    assert update["skills_metadata"][0]["name"] == "quality-review-plugin:review"


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


def test_project_scoped_installs_coexist_across_projects(
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
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    plugin_id = "quality-review-plugin@company-tools"

    monkeypatch.chdir(project_a)
    install_plugin(plugin_id, scope="project")
    monkeypatch.chdir(project_b)
    install_plugin(plugin_id, scope="project")

    entries = load_installed_plugins()[plugin_id]
    assert {entry.project_path for entry in entries} == {
        str(project_a),
        str(project_b),
    }
    monkeypatch.chdir(project_a)
    assert len(discover_plugins().plugins) == 1
    monkeypatch.chdir(project_b)
    assert len(discover_plugins().plugins) == 1


def test_discovery_selects_active_local_entry_after_inactive_project_entry(
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
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    plugin_id = "quality-review-plugin@company-tools"

    monkeypatch.chdir(project_a)
    install_plugin(plugin_id, scope="project")
    monkeypatch.chdir(project_b)
    install_plugin(plugin_id, scope="local")

    entry = get_primary_install_entry(plugin_id, project_path=str(project_b))
    assert entry is not None
    assert entry.scope == "local"
    assert len(discover_plugins().plugins) == 1


def test_invalid_enable_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid plugin install scope"):
        enable_plugin_with_scope("plugin@marketplace", cast("Any", "invalid"))


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
    install_plugin("quality-review-plugin@company-tools", trust=True)
    install_plugin("quality-review-plugin@other-tools", trust=True)

    result = discover_plugins()
    plugins = result.plugins
    mcp_names = {
        name
        for config in plugin_mcp_configs(plugins)
        for name in cast("dict[str, object]", config["mcpServers"])
    }
    skill_prefixes = {prefix for _path, _label, prefix in plugin_skill_sources(plugins)}

    assert len(mcp_names) == 1
    assert len(skill_prefixes) == 1
    assert any("Plugin namespace" in warning for warning in result.warnings)


def test_marketplace_credentials_are_rejected_or_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(MarketplaceError, match="embedded credentials"):
        parse_marketplace_source("https://user:secret@example.com/marketplace.json")

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
    assert "secret" not in stored
    assert "token=%2A%2A%2A" in stored
    assert "channel=stable" in stored

    add_marketplace_source("https://example.com/token/path-credential/marketplace.json")

    stored = load_marketplace_records()["company-tools"].source
    assert "path-credential" not in stored
    assert "/token/***/marketplace.json" in stored


def test_enable_all_reports_failed_installs_without_enabling_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    marketplace_root = tmp_path / "marketplace"
    _make_marketplace(marketplace_root)
    _add_docs_helper_plugin(marketplace_root)
    marketplace = load_marketplace(marketplace_root)
    monkeypatch.setattr(
        "deepagents_code.plugins.commands_cli.add_marketplace_source",
        lambda _source: marketplace,
    )

    def install(plugin_id: str, *, scope: InstallScope) -> None:
        assert scope == "user"
        if plugin_id.startswith("docs-helper@"):
            msg = "unsupported source"
            raise MarketplaceError(msg)

    monkeypatch.setattr("deepagents_code.plugins.commands_cli.install_plugin", install)
    args = argparse.Namespace(
        plugin_command="marketplace",
        marketplace_command="add",
        source="marketplace",
        enable_all=True,
        output_format="text",
    )

    output = execute_plugin_command(args)

    assert output is not None
    assert "Installed: quality-review-plugin@company-tools" in output
    assert "Failed to install: docs-helper@company-tools" in output
    assert "docs-helper@company-tools" not in load_enabled_plugins()


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


def test_plugins_queue_only_bare_manager_command() -> None:
    app = DeepAgentsApp()

    assert app._can_bypass_queue("/plugins") is True
    assert app._can_bypass_queue("/plugins install plugin@marketplace") is False


async def test_plugin_manager_renders_bracketed_errors_as_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepagents_code.tui.widgets.plugin_manager._load_manager_state",
        lambda *_args, **_kwargs: _ManagerState(
            available_plugins=(),
            installed_plugins=(),
            mcp_servers=(),
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


def test_experimental_flag_defaults_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code._env_vars import experimental_enabled

    monkeypatch.delenv(EXPERIMENTAL, raising=False)
    assert experimental_enabled() is False
    monkeypatch.setenv(EXPERIMENTAL, "1")
    assert experimental_enabled() is True
