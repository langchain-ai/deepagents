"""Tests for the MCP disabled-servers persistence store."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from deepagents_code import mcp_disabled
from deepagents_code.configuration.service import invalidate_config_sources
from deepagents_code.mcp_disabled import (
    get_disabled_servers,
    is_server_disabled,
    set_server_disabled,
)
from unit_tests.conftest import redirect_managed_config


@pytest.fixture
def managed_and_user_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, Path]]:
    """Point the default user config and managed config at tmp files."""
    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    invalidate_config_sources()
    yield user, managed
    invalidate_config_sources()


class TestGetDisabledServers:
    """Tests for `get_disabled_servers`."""

    def test_empty_folded_key_shadows_legacy(self, tmp_path: Path) -> None:
        # An empty (but present) folded list is authoritative: once the new
        # shape exists it is the source of truth, so legacy is not consulted.
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[mcp]\ndisabled_servers = []\n[mcp_disabled]\nservers = ["slack"]\n'
        )
        assert get_disabled_servers(config_path=cfg) == set()

    def test_malformed_folded_key_falls_back_to_legacy(self, tmp_path: Path) -> None:
        # A wrong-typed folded value is treated as "unset" (not "empty"), so the
        # legacy list still applies. This is a best-effort convenience list, not
        # a security deny list, so falling back rather than failing closed is fine.
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[mcp]\ndisabled_servers = "github"\n[mcp_disabled]\nservers = ["slack"]\n'
        )
        assert get_disabled_servers(config_path=cfg) == {"slack"}

    def test_returns_empty_on_corrupt_toml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text("this is not valid toml = = =\n")
        assert get_disabled_servers(config_path=cfg) == set()

    def test_managed_denies_survive_corrupt_user_toml(
        self, managed_and_user_configs: tuple[Path, Path]
    ) -> None:
        # A user breaking their own config must not re-enable admin-denied servers.
        user, managed = managed_and_user_configs
        user.write_text("this is not valid toml = = =\n")
        managed.write_text('[mcp]\ndisabled_servers = ["sensitive-server"]\n')
        assert get_disabled_servers() == {"sensitive-server"}


class TestSetServerDisabled:
    """Tests for `set_server_disabled`."""

    def test_refuses_to_overwrite_corrupt_config(self, tmp_path: Path) -> None:
        """Corrupt config must not be silently overwritten.

        A transient parse failure could otherwise truncate sibling
        sections (e.g. model profiles) the next time the user toggles a
        disable state.
        """
        cfg = tmp_path / "config.toml"
        corrupt = "this is not valid toml = = =\n"
        cfg.write_text(corrupt)
        ok, detail = set_server_disabled("github", True, config_path=cfg)
        assert not ok
        assert detail is not None
        # File contents preserved verbatim.
        assert cfg.read_text() == corrupt


class TestIsServerDisabled:
    """Tests for `is_server_disabled`."""

    def test_returns_false_on_corrupt_toml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text("this is not valid toml = = =\n")
        assert not is_server_disabled("github", config_path=cfg)
