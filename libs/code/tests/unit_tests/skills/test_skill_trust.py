"""Tests for the skill directory trust store."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deepagents_code.skills.trust import (
    RevokeResult,
    clear_trusted_skill_dirs,
    is_skill_dir_trusted,
    list_trusted_skill_dir_entries,
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

    def test_trust_persists_approved_path_without_resolving_again(
        self, tmp_path: Path
    ) -> None:
        """Trust stores the approved path even after a symlink swap."""
        store = tmp_path / "skill_trust.json"
        approved = tmp_path / "approved"
        approved.mkdir()
        approved_key = approved.resolve()

        attacker = tmp_path / "attacker"
        attacker.mkdir()
        approved.rmdir()
        approved.symlink_to(attacker)

        assert trust_skill_dir(approved_key, store_path=store)
        assert list_trusted_skill_dirs(store_path=store) == [str(approved_key)]
        assert not is_skill_dir_trusted(attacker, store_path=store)
        assert load_trusted_skill_dirs(store_path=store) == []

    def test_revoke(self, tmp_path: Path) -> None:
        """Revoking trust makes the directory untrusted again."""
        store = tmp_path / "skill_trust.json"
        target = tmp_path / "shared"
        target.mkdir()
        trust_skill_dir(target, store_path=store)
        assert revoke_skill_dir_trust(target, store_path=store) is RevokeResult.REMOVED
        assert not is_skill_dir_trusted(target, store_path=store)

    def test_revoke_nonexistent(self, tmp_path: Path) -> None:
        """Revoking an untrusted directory reports NOT_FOUND, not a false success."""
        store = tmp_path / "skill_trust.json"
        assert (
            revoke_skill_dir_trust(tmp_path / "nope", store_path=store)
            is RevokeResult.NOT_FOUND
        )

    def test_revoke_stale_entry_after_symlink_swap(self, tmp_path: Path) -> None:
        """Revoking the listed path removes stale trust after a symlink swap."""
        store = tmp_path / "skill_trust.json"
        approved = tmp_path / "approved"
        approved.mkdir()
        trust_skill_dir(approved, store_path=store)
        listed = list_trusted_skill_dirs(store_path=store)

        attacker = tmp_path / "attacker"
        attacker.mkdir()
        approved.rmdir()
        approved.symlink_to(attacker)

        assert (
            revoke_skill_dir_trust(listed[0], store_path=store) is RevokeResult.REMOVED
        )
        assert list_trusted_skill_dirs(store_path=store) == []

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

    def test_load_skips_post_approval_symlink_swap(self, tmp_path: Path) -> None:
        """A stored dir swapped for a symlink after approval is not loaded.

        The stored entry is the canonical directory that was approved. If that
        path is later replaced by a symlink to a different directory, loading it
        must not follow the symlink and allowlist the swapped target.
        """
        store = tmp_path / "skill_trust.json"
        approved = tmp_path / "approved"
        approved.mkdir()
        trust_skill_dir(approved, store_path=store)
        assert load_trusted_skill_dirs(store_path=store) == [approved.resolve()]

        # Attacker replaces the approved directory with a symlink elsewhere.
        attacker = tmp_path / "attacker"
        attacker.mkdir()
        approved.rmdir()
        approved.symlink_to(attacker)

        assert load_trusted_skill_dirs(store_path=store) == []

    def test_load_keeps_unchanged_dir(self, tmp_path: Path) -> None:
        """An unchanged stored dir is still returned as its canonical path."""
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

    def test_clear_replaces_corrupt_store(self, tmp_path: Path) -> None:
        """Clearing resets an existing corrupt store."""
        store = tmp_path / "skill_trust.json"
        store.write_text("{not valid json")

        assert clear_trusted_skill_dirs(store_path=store)
        assert list_trusted_skill_dirs(store_path=store, strict=True) == []

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


class TestSkillTrustStoreRobustness:
    """Durability and honesty guarantees for the skill trust store."""

    def test_save_failure_returns_false(self, tmp_path: Path) -> None:
        """An unwritable store path returns False instead of raising."""
        # Parent is a regular file, so mkdir(parents=True) fails with an OSError
        # subclass that _save_store must catch and report as a failed write.
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        store = blocker / "skill_trust.json"
        assert trust_skill_dir(tmp_path, store_path=store) is False

    def test_trust_heals_malformed_dirs_value(self, tmp_path: Path) -> None:
        """A non-dict `dirs` value is replaced, not crashed on or appended to."""
        import json

        store = tmp_path / "skill_trust.json"
        store.write_text(json.dumps({"version": 1, "dirs": []}))
        target = tmp_path / "shared"
        target.mkdir()
        assert trust_skill_dir(target, store_path=store)
        assert is_skill_dir_trusted(target, store_path=store)

    def test_revoke_preserves_other_entries_and_version(self, tmp_path: Path) -> None:
        """Revoking one dir leaves siblings intact and re-stamps the version."""
        import json

        store = tmp_path / "skill_trust.json"
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        trust_skill_dir(a, store_path=store)
        trust_skill_dir(b, store_path=store)
        assert revoke_skill_dir_trust(a, store_path=store) is RevokeResult.REMOVED
        assert not is_skill_dir_trusted(a, store_path=store)
        assert is_skill_dir_trusted(b, store_path=store)
        assert json.loads(store.read_text(encoding="utf-8"))["version"] == 1

    def test_on_disk_shape(self, tmp_path: Path) -> None:
        """The store is a versioned JSON object mapping dirs to metadata."""
        import json

        store = tmp_path / "skill_trust.json"
        target = tmp_path / "shared"
        target.mkdir()
        trust_skill_dir(target, store_path=store)
        data = json.loads(store.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert str(target.resolve()) in data["dirs"]
        assert "trusted_at" in data["dirs"][str(target.resolve())]

    def test_trust_does_not_clobber_on_unreadable_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transient read error aborts the write instead of erasing entries.

        The read/modify/write must not rebuild the store from `{}` on a
        transient `OSError`; doing so would drop every prior approval.
        """
        import deepagents_code.skills.trust as trust_mod

        store = tmp_path / "skill_trust.json"
        first = tmp_path / "first"
        first.mkdir()
        trust_skill_dir(first, store_path=store)
        before = store.read_text(encoding="utf-8")

        monkeypatch.setattr(
            trust_mod, "_load_store", MagicMock(side_effect=OSError("transient"))
        )
        second = tmp_path / "second"
        second.mkdir()
        assert trust_skill_dir(second, store_path=store) is False
        # The original store is untouched — the existing approval survives.
        assert store.read_text(encoding="utf-8") == before

    def test_list_strict_surfaces_unreadable_store(self, tmp_path: Path) -> None:
        """`strict=True` re-raises on a corrupt store; the default degrades."""
        import json

        store = tmp_path / "skill_trust.json"
        store.write_text("{not valid json")
        # Enforcement/default path stays fail-closed (empty).
        assert list_trusted_skill_dirs(store_path=store) == []
        # Audit path opts into surfacing the error.
        with pytest.raises(json.JSONDecodeError):
            list_trusted_skill_dirs(store_path=store, strict=True)

    def test_list_strict_missing_store_is_empty_not_error(self, tmp_path: Path) -> None:
        """A missing store is first-run state, not an error, even under strict."""
        store = tmp_path / "skill_trust.json"
        assert list_trusted_skill_dirs(store_path=store, strict=True) == []

    def test_newer_schema_version_is_refused(self, tmp_path: Path) -> None:
        """A store written by a newer build is not partially read.

        Enforcement/default stays fail-closed (empty) so an unknown schema can't
        grant access by being misread; the audit path surfaces the error.
        """
        import json

        store = tmp_path / "skill_trust.json"
        store.write_text(
            json.dumps({"version": 999, "dirs": {"/shared/a": {}}}),
            encoding="utf-8",
        )
        assert list_trusted_skill_dirs(store_path=store) == []
        with pytest.raises(ValueError, match="unrecognized schema version"):
            list_trusted_skill_dirs(store_path=store, strict=True)

    def test_load_survives_unresolvable_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One entry that fails to resolve is dropped, not fatal to discovery."""
        import json

        store = tmp_path / "skill_trust.json"
        good = tmp_path / "good"
        good.mkdir()
        boom = tmp_path / "boom"
        store.write_text(
            json.dumps(
                {
                    "version": 1,
                    "dirs": {
                        str(good): {"trusted_at": "t"},
                        str(boom): {"trusted_at": "t"},
                    },
                }
            )
        )

        real_resolve = Path.resolve

        def flaky_resolve(self: Path, strict: bool = False) -> Path:
            if self == boom:
                msg = "ELOOP"
                raise OSError(msg)
            return real_resolve(self, strict)

        monkeypatch.setattr(Path, "resolve", flaky_resolve)
        assert load_trusted_skill_dirs(store_path=store) == [good]

    def test_load_survives_entry_that_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `RuntimeError` from `resolve()` drops one entry, not all discovery.

        Some Python builds surface a symlink loop as `RuntimeError` rather than
        `OSError`; the per-entry guard must catch it so one bad stored entry
        can't abort discovery of every other skill.
        """
        import json

        store = tmp_path / "skill_trust.json"
        good = tmp_path / "good"
        good.mkdir()
        boom = tmp_path / "boom"
        store.write_text(
            json.dumps(
                {
                    "version": 1,
                    "dirs": {
                        str(good): {"trusted_at": "t"},
                        str(boom): {"trusted_at": "t"},
                    },
                }
            )
        )

        real_resolve = Path.resolve

        def flaky_resolve(self: Path, strict: bool = False) -> Path:
            if self == boom:
                msg = "symlink loop"
                raise RuntimeError(msg)
            return real_resolve(self, strict)

        monkeypatch.setattr(Path, "resolve", flaky_resolve)
        assert load_trusted_skill_dirs(store_path=store) == [good]

    def test_revoke_does_not_clobber_on_unreadable_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transient read error aborts the revoke instead of erasing entries.

        Mirrors `test_trust_does_not_clobber_on_unreadable_store` for the revoke
        path: a strict-read failure must map to `ERROR` and leave the store
        byte-for-byte unchanged, never rebuild it from `{}` and drop siblings.
        """
        import deepagents_code.skills.trust as trust_mod

        store = tmp_path / "skill_trust.json"
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        trust_skill_dir(a, store_path=store)
        trust_skill_dir(b, store_path=store)
        before = store.read_text(encoding="utf-8")

        monkeypatch.setattr(
            trust_mod, "_load_store", MagicMock(side_effect=OSError("transient"))
        )
        assert revoke_skill_dir_trust(a, store_path=store) is RevokeResult.ERROR
        # Both approvals survive: the store was not rebuilt from an empty dict.
        assert store.read_text(encoding="utf-8") == before

    def test_revoke_save_failure_maps_to_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed write returns `ERROR`, not a false `REMOVED`."""
        import deepagents_code.skills.trust as trust_mod

        store = tmp_path / "skill_trust.json"
        a = tmp_path / "a"
        a.mkdir()
        trust_skill_dir(a, store_path=store)

        monkeypatch.setattr(trust_mod, "_save_store", MagicMock(return_value=False))
        assert revoke_skill_dir_trust(a, store_path=store) is RevokeResult.ERROR

    def test_top_level_not_a_dict(self, tmp_path: Path) -> None:
        """A store whose top-level JSON is not an object is refused/degraded.

        The prior coverage only exercised a non-dict *nested* `dirs`; this pins
        the top-level branch: enforcement degrades to empty, audit surfaces it.
        """
        store = tmp_path / "skill_trust.json"
        store.write_text("[]", encoding="utf-8")
        # Enforcement/default path stays fail-closed (empty).
        assert list_trusted_skill_dirs(store_path=store) == []
        # Audit path opts into surfacing the error.
        with pytest.raises(ValueError, match="not a JSON object"):
            list_trusted_skill_dirs(store_path=store, strict=True)

    def test_non_integer_schema_version_is_refused(self, tmp_path: Path) -> None:
        """A present-but-non-int `version` is unrecognized, not silently trusted.

        Only tampering or a corrupt write produces a non-int version (every
        writer stamps an int), so it must fail closed like a too-new version
        rather than falling through and reading `dirs`.
        """
        import json

        store = tmp_path / "skill_trust.json"
        store.write_text(
            json.dumps({"version": "1", "dirs": {"/shared/a": {}}}),
            encoding="utf-8",
        )
        assert list_trusted_skill_dirs(store_path=store) == []
        with pytest.raises(ValueError, match="unrecognized schema version"):
            list_trusted_skill_dirs(store_path=store, strict=True)

    def test_trust_warns_on_non_canonical_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Trusting a non-canonical path warns at the write boundary.

        Such a key is dropped later by `load_trusted_skill_dirs`' resolve-to-self
        check, so the warning surfaces the caller bug here rather than as a
        silently never-remembered trust.
        """
        import logging

        store = tmp_path / "skill_trust.json"
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.skills.trust"):
            # `link` expanduser()s to itself but resolve()s to `real`, so it is
            # non-canonical and should trip the boundary warning.
            assert trust_skill_dir(link, store_path=store)
        assert any("non-canonical" in r.message for r in caplog.records)

    def test_list_entries_surfaces_trusted_at(self, tmp_path: Path) -> None:
        """`list_trusted_skill_dir_entries` pairs each path with its timestamp."""
        store = tmp_path / "skill_trust.json"
        target = tmp_path / "shared"
        target.mkdir()
        trust_skill_dir(target, store_path=store)

        entries = list_trusted_skill_dir_entries(store_path=store)
        assert len(entries) == 1
        path, trusted_at = entries[0]
        assert path == str(target.resolve())
        # A real ISO-8601 timestamp was recorded, not an empty placeholder.
        assert trusted_at
        from datetime import datetime

        datetime.fromisoformat(trusted_at)  # parses without raising

    def test_load_skips_parent_component_symlink_swap(self, tmp_path: Path) -> None:
        """A swapped *parent* of a stored dir drops the entry, like a leaf swap.

        Both the module docstring and `load_trusted_skill_dirs` claim the
        `resolve()`-to-self check catches a symlink introduced at *any* path
        component, not just the leaf. Replace a parent directory with a symlink
        so the stored path still exists but resolves elsewhere, and confirm the
        entry is dropped rather than followed to the swapped target.
        """
        import shutil

        store = tmp_path / "skill_trust.json"
        skill = tmp_path / "a" / "b" / "skill"
        skill.mkdir(parents=True)
        stored = skill.resolve()
        trust_skill_dir(stored, store_path=store)
        assert load_trusted_skill_dirs(store_path=store) == [stored]

        # Replace the parent component `a/b` with a symlink to a sibling that
        # also contains `skill`, so `stored` remains reachable but canonicalizes
        # to a directory the user never approved.
        evil_parent = tmp_path / "evil"
        (evil_parent / "skill").mkdir(parents=True)
        parent = tmp_path / "a" / "b"
        shutil.rmtree(parent)
        parent.symlink_to(evil_parent, target_is_directory=True)

        assert skill.exists()  # still reachable through the swapped parent
        assert load_trusted_skill_dirs(store_path=store) == []

    def test_load_corrupt_store_fails_closed(self, tmp_path: Path) -> None:
        """`load_trusted_skill_dirs` degrades to empty on a corrupt store.

        The existing corrupt-store tests assert on `list_trusted_skill_dirs`;
        this pins fail-closed at the actual allowlist builder that
        `discover_skills_and_roots` consumes.
        """
        store = tmp_path / "skill_trust.json"
        store.write_text("{not valid json", encoding="utf-8")
        assert load_trusted_skill_dirs(store_path=store) == []

    def test_load_newer_version_fails_closed(self, tmp_path: Path) -> None:
        """A newer-schema store yields no trusted dirs at the enforcement entry.

        A store written by a newer build must not be partially read into the
        containment allowlist; `load_trusted_skill_dirs` (non-strict) returns
        empty rather than trusting `dirs` it may misinterpret.
        """
        import json

        store = tmp_path / "skill_trust.json"
        store.write_text(
            json.dumps({"version": 999, "dirs": {str(tmp_path): {"trusted_at": "t"}}}),
            encoding="utf-8",
        )
        assert load_trusted_skill_dirs(store_path=store) == []
