"""Default marketplace registration and opt-out behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from deepagents_code.plugins.discovery import (
    check_plugin_updates,
    discover_plugins,
    list_available_plugins,
    remove_marketplace,
)
from deepagents_code.plugins.models import MarketplaceRecord
from deepagents_code.plugins.store import (
    load_enabled_plugin_ids,
    load_installed_plugins,
    load_marketplace_records,
    save_marketplace_record,
)
from deepagents_code.tui.modals.plugin_manager.state import _load_manager_state

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )


def test_default_registration_and_startup_are_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    materialize = Mock(side_effect=AssertionError("unexpected network access"))
    monkeypatch.setattr(
        "deepagents_code.plugins.discovery.materialize_marketplace_source", materialize
    )

    record = load_marketplace_records()["langchain-plugins"]
    assert record.source == "langchain-ai/langchain-plugins"
    assert record.source_type == "github"
    assert not discover_plugins().plugins
    assert not check_plugin_updates().errors
    assert not load_installed_plugins()
    assert not load_enabled_plugin_ids()
    assert not (tmp_path / "state" / "plugin_marketplaces.json").exists()
    materialize.assert_not_called()


def test_browsing_fetches_only_missing_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    from deepagents_code.plugins import discovery

    materialize = Mock(wraps=discovery.materialize_marketplace_source)
    monkeypatch.setattr(discovery, "materialize_marketplace_source", materialize)
    state = _load_manager_state()
    assert not state.errors
    assert state.marketplaces[0].name == "langchain-plugins"
    assert state.available_plugins[0].plugin_id == "docs@langchain-plugins"
    assert list_available_plugins() == (("docs@langchain-plugins", "", False),)
    assert _load_manager_state().marketplaces == state.marketplaces
    assert materialize.call_count == 1
    assert not load_installed_plugins()
    assert not load_enabled_plugin_ids()


def test_offline_browsing_does_not_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_CODE_OFFLINE", "1")
    materialize = Mock(side_effect=AssertionError("unexpected network access"))
    monkeypatch.setattr(
        "deepagents_code.plugins.discovery.materialize_marketplace_source", materialize
    )
    assert _load_manager_state().errors
    materialize.assert_not_called()


def test_default_removal_persists_and_can_be_readded() -> None:
    record = load_marketplace_records()["langchain-plugins"]
    assert remove_marketplace("langchain-plugins")
    assert "langchain-plugins" not in load_marketplace_records()
    assert not remove_marketplace("langchain-plugins")
    save_marketplace_record(record)
    assert load_marketplace_records()["langchain-plugins"] == record


def test_existing_marketplace_overrides_default() -> None:
    record = MarketplaceRecord(
        name="langchain-plugins",
        source_type="directory",
        source="/custom/catalog",
        install_location="/custom/catalog",
    )
    save_marketplace_record(record)
    assert load_marketplace_records()["langchain-plugins"] == record
