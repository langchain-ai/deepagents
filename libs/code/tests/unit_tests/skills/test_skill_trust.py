"""Tests for the skill directory trust store."""

from pathlib import Path

import pytest

from deepagents_code.skills.trust import (
    clear_trusted_skill_dirs,
    is_skill_dir_trusted,
    list_trusted_skill_dirs,
    load_trusted_skill_dirs,
    revoke_skill_dir_trust,
    trust_skill_dir,
)


class TestSkillTrustStore:
    """CRUD behavior for the persistent skill trust store."""

    def test_untrusted_by_default(self, tmp_path: Path) -> None:
        """A directory is untrusted when the store file does not exist."""
        store = tmp_path / "skill_trust.json"
        assert not is_skill_dir_trusted(tmp_path / "a", store_path=store)

    def test_trust_and_verify(self, tmp_path: Path) -> None:
        """Trusting a directory then checking returns True."""
        store = tmp_path / "skill_trust.json"
        target = tmp_path / "shared"
        target.mkdir()
        assert trust_skill_dir(target, store_path=store)
        assert is_skill_dir_trusted(target, store_path=store)

    def test_trust_resolves_symlinks(self, tmp_path: Path) -> None:
        """Trust is keyed by the resolved target, so a symlink and its target match."""
        store = tmp_path / "skill_trust.json"
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        trust_skill_dir(link, store_path=store)
        assert is_skill_dir_trusted(real, store_path=store)

    def test_revoke(self, tmp_path: Path) -> None:
        """Revoking trust makes the directory untrusted again."""
        store = tmp_path / "skill_trust.json"
        target = tmp_path / "shared"
        target.mkdir()
        trust_skill_dir(target, store_path=store)
        assert revoke_skill_dir_trust(target, store_path=store)
        assert not is_skill_dir_trusted(target, store_path=store)

    def test_revoke_nonexistent(self, tmp_path: Path) -> None:
        """Revoking an untrusted directory returns True."""
        store = tmp_path / "skill_trust.json"
        assert revoke_skill_dir_trust(tmp_path / "nope", store_path=store)

    def test_list_sorted(self, tmp_path: Path) -> None:
        """Listing returns resolved paths in sorted order."""
        store = tmp_path / "skill_trust.json"
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        trust_skill_dir(b, store_path=store)
        trust_skill_dir(a, store_path=store)
        assert list_trusted_skill_dirs(store_path=store) == sorted(
            [str(a.resolve()), str(b.resolve())]
        )

    def test_load_returns_paths(self, tmp_path: Path) -> None:
        """load_trusted_skill_dirs returns resolved Path objects."""
        store = tmp_path / "skill_trust.json"
        target = tmp_path / "shared"
        target.mkdir()
        trust_skill_dir(target, store_path=store)
        assert load_trusted_skill_dirs(store_path=store) == [target.resolve()]

    def test_clear(self, tmp_path: Path) -> None:
        """Clearing removes every trusted directory."""
        store = tmp_path / "skill_trust.json"
        target = tmp_path / "shared"
        target.mkdir()
        trust_skill_dir(target, store_path=store)
        assert clear_trusted_skill_dirs(store_path=store)
        assert list_trusted_skill_dirs(store_path=store) == []

    def test_corrupt_store_degrades_to_empty(self, tmp_path: Path) -> None:
        """A corrupt store file is treated as nothing trusted."""
        store = tmp_path / "skill_trust.json"
        store.write_text("{not valid json")
        assert list_trusted_skill_dirs(store_path=store) == []
        assert not is_skill_dir_trusted(tmp_path, store_path=store)

    def test_default_store_path_uses_state_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default store path lives under DEFAULT_STATE_DIR."""
        import deepagents_code.model_config as mc

        monkeypatch.setattr(mc, "DEFAULT_STATE_DIR", tmp_path)
        target = tmp_path / "shared"
        target.mkdir()
        assert trust_skill_dir(target)
        assert (tmp_path / "skill_trust.json").exists()
        assert is_skill_dir_trusted(target)
