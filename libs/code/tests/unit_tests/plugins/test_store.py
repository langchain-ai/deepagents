from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_code.plugins.models import MarketplaceRecord
from deepagents_code.plugins.store import (
    PluginStateError,
    add_installed_plugin,
    save_marketplace_record,
    set_plugin_enabled,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "state"
    path.mkdir()
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", path)
    return path


def test_marketplace_mutation_preserves_corrupt_state(state_dir: Path) -> None:
    path = state_dir / "plugin_marketplaces.json"
    original = "{not valid json"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(PluginStateError, match="could not be read"):
        save_marketplace_record(
            MarketplaceRecord(
                name="tools",
                source_type="github",
                source="owner/repo",
                install_location="/cache/tools",
            )
        )

    assert path.read_text(encoding="utf-8") == original
