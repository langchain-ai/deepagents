"""Offline marketplace fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_code.plugins import marketplace

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import MarketplaceSource, PluginMarketplace


@pytest.fixture(autouse=True)
def offline_default_marketplace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep default catalog discovery independent of GitHub."""
    monkeypatch.delenv("DEEPAGENTS_CODE_OFFLINE", raising=False)
    materialize = marketplace.materialize_marketplace_source
    root = tmp_path / "default-marketplace"
    root.mkdir()
    (root / "marketplace.json").write_text(
        '{"name": "langchain-plugins", "plugins": [{"name": "docs", '
        '"source": {"source": "github", "repo": "example/docs"}}]}',
        encoding="utf-8",
    )

    def load(source: MarketplaceSource) -> tuple[PluginMarketplace, Path]:
        if (
            source.source_type == "github"
            and source.value == "langchain-ai/langchain-plugins"
        ):
            return marketplace.load_marketplace_location(
                root / "marketplace.json"
            ), root / "marketplace.json"
        return materialize(source)

    monkeypatch.setattr(
        "deepagents_code.plugins.discovery.materialize_marketplace_source", load
    )
