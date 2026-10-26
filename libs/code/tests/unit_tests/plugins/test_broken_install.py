"""Enabled plugins whose installed files are missing."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deepagents_code.plugins import (
    add_local_marketplace,
    discover_plugins,
    install_plugin,
    reinstall_plugin,
    set_installed_plugin_enabled,
)
from deepagents_code.plugins.adapters.hooks import discover_plugin_hook_sources
from deepagents_code.plugins.adapters.mcp import discover_plugin_mcp_configs
from deepagents_code.plugins.commands_cli import execute_plugin_command
from deepagents_code.plugins.discovery import claim_broken_install_notices
from deepagents_code.plugins.models import BrokenPluginInstall
from deepagents_code.plugins.store import (
    get_primary_install_entry,
    load_enabled_plugin_ids,
    plugin_data_dir,
)
from deepagents_code.tui.modals.plugin_manager.content import (
    _installed_details_options,
    _status_lines,
)
from deepagents_code.tui.modals.plugin_manager.state import _load_manager_state

MARKETPLACE = "company-tools"
BROKEN = f"broken-plugin@{MARKETPLACE}"
HEALTHY = f"healthy-plugin@{MARKETPLACE}"


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_plugin(root: Path, name: str) -> None:
    _write_json(root / ".claude-plugin" / "plugin.json", {"name": name, "version": "1"})
    skill = root / "skills" / "review" / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        "---\nname: review\ndescription: Review code.\n---\n\nReview.",
        encoding="utf-8",
    )
    _write_json(root / ".mcp.json", {"mcpServers": {"docs": {"command": "docs"}}})
    _write_json(
        root / "hooks" / "hooks.json",
        {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "x"}]}]}},
    )


@pytest.fixture
def plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install a broken and a healthy plugin in isolated state directories.

    Returns:
        The broken plugin's registered install path, already deleted.
    """
    monkeypatch.delenv("DEEPAGENTS_CODE_PLUGIN_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    monkeypatch.setattr("deepagents_code.plugins.discovery._reported_broken", set())
    root = tmp_path / "marketplace"
    _write_json(
        root / ".claude-plugin" / "marketplace.json",
        {
            "name": MARKETPLACE,
            "owner": {"name": "Team"},
            "plugins": [
                {"name": "broken-plugin", "source": "./plugins/broken-plugin"},
                {"name": "healthy-plugin", "source": "./plugins/healthy-plugin"},
            ],
        },
    )
    _make_plugin(root / "plugins" / "broken-plugin", "broken-plugin")
    _make_plugin(root / "plugins" / "healthy-plugin", "healthy-plugin")
    add_local_marketplace(root)
    install_plugin(BROKEN)
    install_plugin(HEALTHY)
    entry = get_primary_install_entry(BROKEN)
    assert entry is not None
    shutil.rmtree(entry.install_path)
    return Path(entry.install_path)


def test_discovery_reports_missing_install_and_loads_sibling(plugins: Path) -> None:
    result = discover_plugins()

    assert [plugin.plugin_id for plugin in result.plugins] == [HEALTHY]
    assert result.broken == (
        BrokenPluginInstall(plugin_id=BROKEN, install_path=str(plugins)),
    )
    assert result.warnings == ()


def test_broken_install_message_is_actionable_without_internals(
    plugins: Path,
) -> None:
    (item,) = discover_plugins().broken

    assert item.message == (
        f"Plugin {BROKEN} needs repair: its installed files are missing, so its "
        "tools are unavailable. Reinstall or disable it from /plugins."
    )
    assert str(plugins) not in item.message
    assert "cache" not in item.message


def test_consumers_keep_path_in_debug_logs_only(
    plugins: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="deepagents_code")

    configs = discover_plugin_mcp_configs()
    _documents, diagnostics = discover_plugin_hook_sources()

    servers = [
        name
        for config in configs
        if isinstance(layer := config["mcpServers"], dict)
        for name in layer
    ]
    assert len(servers) == 1
    assert "healthy" in servers[0]
    assert diagnostics == ()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert all(BROKEN not in record.getMessage() for record in warnings)
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(str(plugins) in record.getMessage() for record in debug)


@pytest.mark.usefixtures("plugins")
def test_claim_deduplicates_notices_per_broken_install() -> None:
    broken = discover_plugins().broken
    other = BrokenPluginInstall(plugin_id=f"other@{MARKETPLACE}", install_path="/x")

    assert claim_broken_install_notices(broken) == broken
    assert claim_broken_install_notices(discover_plugins().broken) == ()
    assert claim_broken_install_notices((*broken, other)) == (other,)


@pytest.mark.usefixtures("plugins")
def test_app_notifies_once_across_discoveries() -> None:
    from deepagents_code.app import DeepAgentsApp

    app = DeepAgentsApp()
    notify = MagicMock()
    app.notify = notify  # type: ignore[method-assign]  # capture toasts without mounting

    app._notify_broken_plugins(discover_plugins().broken)
    app._notify_broken_plugins(discover_plugins().broken)

    notify.assert_called_once()
    assert notify.call_args.args[0].startswith(f"Plugin {BROKEN} needs repair")
    assert notify.call_args.kwargs["markup"] is False


@pytest.mark.usefixtures("plugins")
def test_reinstall_restores_files_and_preserves_state() -> None:
    data_file = plugin_data_dir(BROKEN) / "settings.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text('{"token": "keep"}', encoding="utf-8")
    enabled_before = load_enabled_plugin_ids()
    assert claim_broken_install_notices(discover_plugins().broken)

    instance = reinstall_plugin(BROKEN)

    assert (instance.root / "skills" / "review" / "SKILL.md").is_file()
    assert load_enabled_plugin_ids() == enabled_before
    assert data_file.read_text(encoding="utf-8") == '{"token": "keep"}'
    result = discover_plugins()
    assert {plugin.plugin_id for plugin in result.plugins} == {BROKEN, HEALTHY}
    assert result.broken == ()
    shutil.rmtree(instance.root)
    assert claim_broken_install_notices(discover_plugins().broken)


@pytest.mark.usefixtures("plugins")
def test_reinstall_replaces_damaged_copy_of_same_version() -> None:
    instance = reinstall_plugin(BROKEN)
    skill = instance.root / "skills" / "review" / "SKILL.md"
    skill.unlink()

    reinstall_plugin(BROKEN)

    assert skill.is_file()


@pytest.mark.usefixtures("plugins")
def test_reinstall_keeps_disabled_plugin_disabled() -> None:
    set_installed_plugin_enabled(BROKEN, enabled=False)

    reinstall_plugin(BROKEN)

    assert BROKEN not in load_enabled_plugin_ids()
    assert HEALTHY in load_enabled_plugin_ids()


@pytest.mark.usefixtures("plugins")
def test_manager_shows_needs_repair_with_reinstall_action() -> None:
    state = _load_manager_state()
    row = next(r for r in state.installed_plugins if r.plugin_id == BROKEN)
    healthy = next(r for r in state.installed_plugins if r.plugin_id == HEALTHY)

    assert row.load_state == "needs_repair"
    assert healthy.load_state != "needs_repair"
    assert str(_status_lines(row)[0]).startswith("Status: Needs repair")
    ids = [option.id for option in _installed_details_options(row, divider_width=8)]
    assert ids[:2] == ["action:reinstall", "action:toggle-enabled"]
    healthy_ids = [
        option.id for option in _installed_details_options(healthy, divider_width=8)
    ]
    assert "action:reinstall" not in healthy_ids


@pytest.mark.usefixtures("plugins")
def test_cli_lists_needs_repair_and_reinstalls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    listing = execute_plugin_command(argparse.Namespace(plugin_command="list"))
    assert listing is not None
    assert f"needs repair {BROKEN}" in listing
    assert f"enabled {HEALTHY}" in listing

    text = execute_plugin_command(
        argparse.Namespace(plugin_command="reinstall", plugin_id=BROKEN)
    )

    assert text == f"Reinstalled plugin {BROKEN}. Run /reload to activate."
    assert discover_plugins().broken == ()
    capsys.readouterr()
