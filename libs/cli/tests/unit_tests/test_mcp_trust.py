"""Tests for MCP project-level trust store."""

from pathlib import Path

import pytest

from deepagents_cli.mcp_trust import (
    compute_config_fingerprint,
    is_project_mcp_trusted,
    revoke_project_mcp_trust,
    trust_project_mcp,
)


class TestComputeConfigFingerprint:
    """Tests for compute_config_fingerprint."""

    def test_empty_list(self) -> None:
        """Empty path list produces a deterministic hash of empty input."""
        fp = compute_config_fingerprint([])
        assert fp.startswith("sha256:")
        assert len(fp) == len("sha256:") + 64

    def test_deterministic(self, tmp_path: Path) -> None:
        """Same file content produces the same fingerprint."""
        f = tmp_path / "a.json"
        f.write_text('{"mcpServers": {}}')
        assert compute_config_fingerprint([f]) == compute_config_fingerprint([f])

    def test_different_content_different_fingerprint(self, tmp_path: Path) -> None:
        """Different content produces different fingerprints."""
        a = tmp_path / "a.json"
        a.write_text('{"a": 1}')
        b = tmp_path / "b.json"
        b.write_text('{"b": 2}')
        assert compute_config_fingerprint([a]) != compute_config_fingerprint([b])

    def test_sorted_order(self, tmp_path: Path) -> None:
        """Fingerprint is stable regardless of input order."""
        a = tmp_path / "a.json"
        a.write_text("aaa")
        b = tmp_path / "b.json"
        b.write_text("bbb")
        assert compute_config_fingerprint([a, b]) == compute_config_fingerprint([b, a])

    def test_missing_file_does_not_error(self, tmp_path: Path) -> None:
        """Missing paths are skipped gracefully."""
        missing = tmp_path / "nope.json"
        fp = compute_config_fingerprint([missing])
        assert fp.startswith("sha256:")


class TestTrustStore:
    """Tests for is_project_mcp_trusted / trust_project_mcp / revoke."""

    def test_untrusted_by_default(self, tmp_path: Path) -> None:
        """A project is not trusted when the config file doesn't exist."""
        cfg = tmp_path / "config.toml"
        assert not is_project_mcp_trusted(
            "/some/project", "sha256:abc", config_path=cfg
        )

    def test_trust_and_verify(self, tmp_path: Path) -> None:
        """Trusting a project then checking returns True."""
        cfg = tmp_path / "config.toml"
        fp = "sha256:deadbeef"
        assert trust_project_mcp("/my/project", fp, config_path=cfg)
        assert is_project_mcp_trusted("/my/project", fp, config_path=cfg)

    def test_fingerprint_mismatch(self, tmp_path: Path) -> None:
        """Different fingerprint returns False."""
        cfg = tmp_path / "config.toml"
        trust_project_mcp("/my/project", "sha256:aaa", config_path=cfg)
        assert not is_project_mcp_trusted("/my/project", "sha256:bbb", config_path=cfg)

    def test_revoke(self, tmp_path: Path) -> None:
        """Revoking trust makes the project untrusted."""
        cfg = tmp_path / "config.toml"
        fp = "sha256:123"
        trust_project_mcp("/proj", fp, config_path=cfg)
        assert is_project_mcp_trusted("/proj", fp, config_path=cfg)
        assert revoke_project_mcp_trust("/proj", config_path=cfg)
        assert not is_project_mcp_trusted("/proj", fp, config_path=cfg)

    def test_revoke_nonexistent(self, tmp_path: Path) -> None:
        """Revoking a nonexistent entry returns True."""
        cfg = tmp_path / "config.toml"
        assert revoke_project_mcp_trust("/nope", config_path=cfg)

    def test_multiple_projects(self, tmp_path: Path) -> None:
        """Multiple projects can be independently trusted."""
        cfg = tmp_path / "config.toml"
        trust_project_mcp("/a", "sha256:a1", config_path=cfg)
        trust_project_mcp("/b", "sha256:b1", config_path=cfg)
        assert is_project_mcp_trusted("/a", "sha256:a1", config_path=cfg)
        assert is_project_mcp_trusted("/b", "sha256:b1", config_path=cfg)

    def test_preserves_existing_config(self, tmp_path: Path) -> None:
        """Trust operations preserve other config sections."""
        import tomllib

        import tomli_w

        cfg = tmp_path / "config.toml"
        cfg.write_text('[warnings]\nsuppress = ["ripgrep"]\n')

        trust_project_mcp("/proj", "sha256:x", config_path=cfg)

        with cfg.open("rb") as f:
            data = tomllib.load(f)
        assert data["warnings"]["suppress"] == ["ripgrep"]
        assert data["mcp_trust"]["projects"]["/proj"] == "sha256:x"
