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
        cfg.write_text('[mcp_disabled]\nservers = ["github", "slack"]\n')
        assert get_disabled_servers(config_path=cfg) == {"github", "slack"}

    def test_filters_non_string_entries(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[mcp_disabled]\nservers = ["ok", ""]\n')
        assert get_disabled_servers(config_path=cfg) == {"ok"}


class TestSetServerDisabled:
    """Tests for `set_server_disabled`."""

    def test_disable_new_server(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        assert set_server_disabled("github", True, config_path=cfg)
        assert is_server_disabled("github", config_path=cfg)

    def test_disable_is_idempotent(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        set_server_disabled("github", True, config_path=cfg)
        assert set_server_disabled("github", True, config_path=cfg)
        assert get_disabled_servers(config_path=cfg) == {"github"}

    def test_enable_removes_entry(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        set_server_disabled("github", True, config_path=cfg)
        set_server_disabled("slack", True, config_path=cfg)
        assert set_server_disabled("github", False, config_path=cfg)
        assert get_disabled_servers(config_path=cfg) == {"slack"}

    def test_enable_missing_is_noop(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        assert set_server_disabled("nonexistent", False, config_path=cfg)
        assert get_disabled_servers(config_path=cfg) == set()

    def test_preserves_other_sections(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[other]\nkey = "value"\n')
        set_server_disabled("github", True, config_path=cfg)
        contents = cfg.read_text()
        assert "[other]" in contents
        assert 'key = "value"' in contents
        assert "[mcp_disabled]" in contents

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


class TestIsServerDisabled:
    """Tests for `is_server_disabled`."""

    def test_returns_false_when_empty(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        assert not is_server_disabled("github", config_path=cfg)

    def test_returns_true_after_disable(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        set_server_disabled("github", True, config_path=cfg)
        assert is_server_disabled("github", config_path=cfg)
