from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import pytest

from deepagents_code._env_vars import OFFLINE
from deepagents_code.plugins import add_local_marketplace, install_plugin
from deepagents_code.plugins.commands_cli import execute_plugin_command
from deepagents_code.plugins.discovery import check_plugin_updates
from deepagents_code.plugins.marketplace import load_marketplace
from deepagents_code.plugins.models import MarketplaceRecord
from deepagents_code.plugins.store import (
    get_primary_install_entry,
    load_available_plugin_updates,
    save_marketplace_record,
)

if TYPE_CHECKING:
    from pathlib import Path

PLUGIN_ID = "review@tools"


def _write_manifest(root: Path, version: str, *, auto_update: bool = False) -> None:
    manifest: dict[str, object] = {"name": "review", "version": version}
    if auto_update:
        manifest["extensions"] = {"com.langchain.deepagents.code": {"autoUpdate": True}}
    path = root / "plugins" / "review" / ".claude-plugin" / "plugin.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def marketplace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(
        "deepagents_code.plugins.discovery.plugin_auto_update_setting",
        lambda: (True, "default"),
    )
    root = tmp_path / "marketplace"
    marketplace_json = root / ".claude-plugin" / "marketplace.json"
    marketplace_json.parent.mkdir(parents=True)
    marketplace_json.write_text(
        json.dumps(
            {
                "name": "tools",
                "plugins": [{"name": "review", "source": "./plugins/review"}],
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(root, "1.0.0")
    skill = root / "plugins" / "review" / "skills" / "review" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: review\ndescription: Review code.\n---\n\nReview.", encoding="utf-8"
    )
    add_local_marketplace(root)
    install_plugin(PLUGIN_ID)
    return root


def _as_remote(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the local marketplace as a GitHub source served from `root`."""
    save_marketplace_record(
        MarketplaceRecord(
            name="tools",
            source_type="github",
            source="example/tools",
            install_location=str(root),
        )
    )
    monkeypatch.setattr(
        "deepagents_code.plugins.discovery.materialize_marketplace_source",
        lambda _source: (load_marketplace(root), root),
    )


def test_check_reports_update_without_installing_and_notifies_once(
    marketplace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _as_remote(marketplace_root, monkeypatch)
    _write_manifest(marketplace_root, "1.1.0")

    first = check_plugin_updates()
    second = check_plugin_updates()

    assert [u.available_version for u in first.available] == ["1.1.0"]
    assert first.newly_available == first.available
    assert not first.updated
    assert second.available == first.available
    assert not second.newly_available
    assert load_available_plugin_updates() == {PLUGIN_ID: "1.1.0"}
    entry = get_primary_install_entry(PLUGIN_ID)
    assert entry is not None
    assert entry.version == "1.0.0"


def test_check_installs_opted_in_update(
    marketplace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _as_remote(marketplace_root, monkeypatch)
    _write_manifest(marketplace_root, "1.1.0", auto_update=True)

    result = check_plugin_updates()

    assert [u.plugin_id for u in result.updated] == [PLUGIN_ID]
    assert not result.available
    entry = get_primary_install_entry(PLUGIN_ID)
    assert entry is not None
    assert entry.version == "1.1.0"


def test_cli_update_check_then_install(
    marketplace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _as_remote(marketplace_root, monkeypatch)
    _write_manifest(marketplace_root, "1.1.0")
    check_plugin_updates()

    def run(*, check: bool) -> str | None:
        args = argparse.Namespace(
            plugin_command="update",
            plugin_ids=[],
            check=check,
            output_format="text",
        )
        return execute_plugin_command(args)

    assert run(check=True) == f"Update available: {PLUGIN_ID} 1.0.0 -> 1.1.0"
    assert run(check=False) == (
        f"Updated {PLUGIN_ID} 1.0.0 -> 1.1.0.\nRun /reload to apply."
    )
    assert run(check=True) == "All plugins are up to date."
    assert load_available_plugin_updates() == {}


@pytest.mark.usefixtures("marketplace_root")
def test_cli_update_rejects_uninstalled_plugin() -> None:
    args = argparse.Namespace(
        plugin_command="update",
        plugin_ids=["missing@tools"],
        check=False,
        output_format="text",
    )

    with pytest.raises(SystemExit):
        execute_plugin_command(args)


def test_plugin_manager_offers_available_update(
    marketplace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.tui.modals.plugin_manager.content import (
        _installed_details_options,
    )
    from deepagents_code.tui.modals.plugin_manager.state import _load_manager_state

    _as_remote(marketplace_root, monkeypatch)
    _write_manifest(marketplace_root, "1.1.0")
    check_plugin_updates()

    (row,) = _load_manager_state().installed_plugins
    options = _installed_details_options(row, divider_width=10)

    assert row.available_version == "1.1.0"
    assert options[0].id == "action:update"
