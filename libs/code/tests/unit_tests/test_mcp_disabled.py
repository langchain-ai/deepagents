"""Tests for the MCP disabled-servers persistence store."""

from pathlib import Path

from deepagents_code.mcp_disabled import (
    get_disabled_servers,
    is_server_disabled,
    set_server_disabled,
)


class TestGetDisabledServers:
    """Tests for `get_disabled_servers`."""

    def test_empty_when_no_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        assert get_disabled_servers(config_path=cfg) == set()

    def test_empty_when_section_missing(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[other]\nkey = "value"\n')
        assert get_disabled_servers(config_path=cfg) == set()

    def test_reads_existing_entries(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[mcp]\ndisabled_servers = ["github", "slack"]\n')
        assert get_disabled_servers(config_path=cfg) == {"github", "slack"}

    def test_reads_legacy_entries_when_folded_key_missing(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[mcp_disabled]\nservers = ["github", "slack"]\n')
        assert get_disabled_servers(config_path=cfg) == {"github", "slack"}

    def test_folded_key_wins_over_legacy_entries(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[mcp]\ndisabled_servers = ["github"]\n'
            '[mcp_disabled]\nservers = ["slack"]\n'
        )
        assert get_disabled_servers(config_path=cfg) == {"github"}

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

    def test_filters_non_string_entries(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[mcp]\ndisabled_servers = ["ok", ""]\n')
        assert get_disabled_servers(config_path=cfg) == {"ok"}

    def test_returns_empty_on_corrupt_toml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text("this is not valid toml = = =\n")
        assert get_disabled_servers(config_path=cfg) == set()


class TestSetServerDisabled:
    """Tests for `set_server_disabled`."""

    def test_disable_new_server(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        ok, detail = set_server_disabled("github", True, config_path=cfg)
        assert ok
        assert detail is None
        assert is_server_disabled("github", config_path=cfg)

    def test_disable_is_idempotent(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        set_server_disabled("github", True, config_path=cfg)
        ok, _ = set_server_disabled("github", True, config_path=cfg)
        assert ok
        assert get_disabled_servers(config_path=cfg) == {"github"}

    def test_enable_removes_entry(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        set_server_disabled("github", True, config_path=cfg)
        set_server_disabled("slack", True, config_path=cfg)
        ok, _ = set_server_disabled("github", False, config_path=cfg)
        assert ok
        assert get_disabled_servers(config_path=cfg) == {"slack"}

    def test_enable_missing_is_noop(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        ok, _ = set_server_disabled("nonexistent", False, config_path=cfg)
        assert ok
        assert get_disabled_servers(config_path=cfg) == set()

    def test_preserves_other_sections(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[other]\nkey = "value"\n')
        set_server_disabled("github", True, config_path=cfg)
        contents = cfg.read_text()
        assert "[other]" in contents
        assert 'key = "value"' in contents
        assert "[mcp]" in contents
        assert "disabled_servers" in contents
        assert "github" in contents

    def test_preserves_existing_mcp_keys(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[mcp]\nenabled_project_servers = ["docs"]\n')
        ok, detail = set_server_disabled("github", True, config_path=cfg)
        assert ok
        assert detail is None
        contents = cfg.read_text()
        assert "enabled_project_servers" in contents
        assert "docs" in contents
        assert "disabled_servers" in contents
        assert "github" in contents

    def test_migrates_legacy_section_on_write(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[mcp_disabled]\nservers = ["github"]\n')
        ok, detail = set_server_disabled("slack", True, config_path=cfg)
        assert ok
        assert detail is None
        contents = cfg.read_text()
        assert "[mcp_disabled]" not in contents
        assert "[mcp]" in contents
        assert get_disabled_servers(config_path=cfg) == {"github", "slack"}

    def test_entries_sorted(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        set_server_disabled("zeta", True, config_path=cfg)
        set_server_disabled("alpha", True, config_path=cfg)
        set_server_disabled("mango", True, config_path=cfg)
        assert get_disabled_servers(config_path=cfg) == {"alpha", "mango", "zeta"}
        # Confirm on-disk order is alphabetical for diff-friendliness.
        contents = cfg.read_text()
        a_idx = contents.index("alpha")
        m_idx = contents.index("mango")
        z_idx = contents.index("zeta")
        assert a_idx < m_idx < z_idx

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

    def test_returns_false_when_empty(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        assert not is_server_disabled("github", config_path=cfg)

    def test_returns_true_after_disable(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        set_server_disabled("github", True, config_path=cfg)
        assert is_server_disabled("github", config_path=cfg)

    def test_returns_false_on_corrupt_toml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text("this is not valid toml = = =\n")
        assert not is_server_disabled("github", config_path=cfg)
