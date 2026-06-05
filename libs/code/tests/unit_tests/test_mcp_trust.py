"""Tests for MCP project-level trust store."""

from pathlib import Path

import pytest

from deepagents_code.mcp_trust import (
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
        """A project is not trusted when the store file doesn't exist."""
        store = tmp_path / "mcp_trust.json"
        assert not is_project_mcp_trusted(
            "/some/project", "sha256:abc", store_path=store
        )

    def test_trust_and_verify(self, tmp_path: Path) -> None:
        """Trusting a project then checking returns True."""
        store = tmp_path / "mcp_trust.json"
        fp = "sha256:deadbeef"
        assert trust_project_mcp("/my/project", fp, store_path=store)
        assert is_project_mcp_trusted("/my/project", fp, store_path=store)

    def test_fingerprint_mismatch(self, tmp_path: Path) -> None:
        """Different fingerprint returns False."""
        store = tmp_path / "mcp_trust.json"
        trust_project_mcp("/my/project", "sha256:aaa", store_path=store)
        assert not is_project_mcp_trusted("/my/project", "sha256:bbb", store_path=store)

    def test_revoke(self, tmp_path: Path) -> None:
        """Revoking trust makes the project untrusted."""
        store = tmp_path / "mcp_trust.json"
        fp = "sha256:123"
        trust_project_mcp("/proj", fp, store_path=store)
        assert is_project_mcp_trusted("/proj", fp, store_path=store)
        assert revoke_project_mcp_trust("/proj", store_path=store)
        assert not is_project_mcp_trusted("/proj", fp, store_path=store)

    def test_revoke_nonexistent(self, tmp_path: Path) -> None:
        """Revoking a nonexistent entry returns True."""
        store = tmp_path / "mcp_trust.json"
        assert revoke_project_mcp_trust("/nope", store_path=store)

    def test_multiple_projects(self, tmp_path: Path) -> None:
        """Multiple projects can be independently trusted."""
        store = tmp_path / "mcp_trust.json"
        trust_project_mcp("/a", "sha256:a1", store_path=store)
        trust_project_mcp("/b", "sha256:b1", store_path=store)
        assert is_project_mcp_trusted("/a", "sha256:a1", store_path=store)
        assert is_project_mcp_trusted("/b", "sha256:b1", store_path=store)

    def test_save_failure_returns_false(self, tmp_path: Path) -> None:
        """An unwritable store path returns False instead of raising."""
        # Parent is a regular file, so mkdir(parents=True) fails with an OSError
        # subclass that _save_store must catch and report as a failed write.
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        store = blocker / "mcp_trust.json"
        assert trust_project_mcp("/proj", "sha256:x", store_path=store) is False

    def test_trust_heals_malformed_projects_value(self, tmp_path: Path) -> None:
        """A non-dict `projects` value is replaced, not crashed on or appended to."""
        import json

        store = tmp_path / "mcp_trust.json"
        store.write_text(json.dumps({"version": 1, "projects": []}))
        assert trust_project_mcp("/proj", "sha256:x", store_path=store)
        assert is_project_mcp_trusted("/proj", "sha256:x", store_path=store)

    def test_revoke_preserves_other_entries(self, tmp_path: Path) -> None:
        """Revoking one project leaves siblings intact and re-stamps the version."""
        import json

        store = tmp_path / "mcp_trust.json"
        trust_project_mcp("/a", "sha256:a1", store_path=store)
        trust_project_mcp("/b", "sha256:b1", store_path=store)
        assert revoke_project_mcp_trust("/a", store_path=store)
        assert not is_project_mcp_trusted("/a", "sha256:a1", store_path=store)
        assert is_project_mcp_trusted("/b", "sha256:b1", store_path=store)
        data = json.loads(store.read_text(encoding="utf-8"))
        assert data["version"] == 1

    def test_on_disk_shape(self, tmp_path: Path) -> None:
        """The store is a versioned JSON object mapping roots to fingerprints."""
        import json

        store = tmp_path / "mcp_trust.json"
        trust_project_mcp("/proj", "sha256:x", store_path=store)

        data = json.loads(store.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["projects"]["/proj"] == "sha256:x"

    def test_corrupt_store_degrades_to_untrusted(self, tmp_path: Path) -> None:
        """A corrupt store reads as nothing-trusted instead of raising."""
        store = tmp_path / "mcp_trust.json"
        store.write_text("not valid json {{{")
        assert not is_project_mcp_trusted("/proj", "sha256:x", store_path=store)
        # A subsequent write rewrites the file cleanly.
        assert trust_project_mcp("/proj", "sha256:x", store_path=store)
        assert is_project_mcp_trusted("/proj", "sha256:x", store_path=store)

    def test_default_path_uses_state_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an explicit path, the store lives under DEFAULT_STATE_DIR."""
        from deepagents_code import model_config

        monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", tmp_path)
        trust_project_mcp("/proj", "sha256:x")
        assert (tmp_path / "mcp_trust.json").exists()
        assert is_project_mcp_trusted("/proj", "sha256:x")
