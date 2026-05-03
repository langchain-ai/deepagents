"""Tests for the legacy-state migration helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_cli.state_migration import migrate_legacy_state

if TYPE_CHECKING:
    import pytest


def _snapshot(state_dir: Path) -> list[tuple[str, bytes | None]]:
    """Return a sorted (relpath, content) snapshot of `state_dir`."""
    items: list[tuple[str, bytes | None]] = []
    for entry in state_dir.rglob("*"):
        rel = str(entry.relative_to(state_dir))
        items.append((rel, entry.read_bytes() if entry.is_file() else None))
    return sorted(items)


def _seed_legacy(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "sessions.db").write_bytes(b"db")
    (config_dir / "history.jsonl").write_text("h\n")
    (config_dir / "latest_version.json").write_text(json.dumps({"v": "1"}))
    (config_dir / "onboarding_complete").write_text("1\n")
    tokens = config_dir / "mcp-tokens"
    tokens.mkdir()
    (tokens / "notion-x.json").write_text("{}")


class TestMigrateLegacyState:
    """Behaviour of `migrate_legacy_state`."""

    def test_moves_files_into_state_dir(self, tmp_path: Path) -> None:
        """Each legacy entry is moved under `.state/`."""
        config_dir = tmp_path / ".deepagents"
        state_dir = config_dir / ".state"
        _seed_legacy(config_dir)

        migrate_legacy_state(config_dir=config_dir, state_dir=state_dir)

        assert (state_dir / "sessions.db").read_bytes() == b"db"
        assert (state_dir / "history.jsonl").read_text() == "h\n"
        assert json.loads((state_dir / "latest_version.json").read_text()) == {"v": "1"}
        assert (state_dir / "mcp-tokens" / "notion-x.json").exists()
        assert (state_dir / "onboarding_complete").read_text() == "1\n"

        for legacy_name in (
            "sessions.db",
            "history.jsonl",
            "latest_version.json",
            "mcp-tokens",
            "onboarding_complete",
        ):
            assert not (config_dir / legacy_name).exists(), legacy_name

    def test_idempotent(self, tmp_path: Path) -> None:
        """Re-running with no legacy left in place does nothing."""
        config_dir = tmp_path / ".deepagents"
        state_dir = config_dir / ".state"
        _seed_legacy(config_dir)

        migrate_legacy_state(config_dir=config_dir, state_dir=state_dir)
        before = _snapshot(state_dir)
        # Second run must be a no-op.
        migrate_legacy_state(config_dir=config_dir, state_dir=state_dir)
        assert _snapshot(state_dir) == before

    def test_skips_when_dest_already_exists(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If a file already exists at dest, legacy is left alone with a warning."""
        config_dir = tmp_path / ".deepagents"
        state_dir = config_dir / ".state"
        config_dir.mkdir()
        state_dir.mkdir()
        (config_dir / "sessions.db").write_bytes(b"old")
        (state_dir / "sessions.db").write_bytes(b"new")

        with caplog.at_level("WARNING", logger="deepagents_cli.state_migration"):
            migrate_legacy_state(config_dir=config_dir, state_dir=state_dir)

        # Destination preserved; legacy not clobbered, not moved.
        assert (state_dir / "sessions.db").read_bytes() == b"new"
        assert (config_dir / "sessions.db").read_bytes() == b"old"
        # The collision is loud — users hitting this scenario need to see it.
        assert any(
            "destination already exists" in record.getMessage()
            for record in caplog.records
        )

    def test_no_legacy_no_state_dir_created(self, tmp_path: Path) -> None:
        """When nothing needs moving, the state dir isn't created."""
        config_dir = tmp_path / ".deepagents"
        state_dir = config_dir / ".state"
        config_dir.mkdir()

        migrate_legacy_state(config_dir=config_dir, state_dir=state_dir)

        assert not state_dir.exists()

    def test_missing_config_dir_is_no_op(self, tmp_path: Path) -> None:
        """Fresh installs (no `~/.deepagents/`) are handled silently."""
        config_dir = tmp_path / ".deepagents"
        state_dir = config_dir / ".state"

        migrate_legacy_state(config_dir=config_dir, state_dir=state_dir)

        assert not config_dir.exists()
        assert not state_dir.exists()

    def test_sqlite_sidecars_migrate_with_main_db(self, tmp_path: Path) -> None:
        """`sessions.db-wal` and `-shm` move alongside `sessions.db`.

        Splitting the main DB from its WAL would corrupt the database on
        next open, so the sidecars must travel as a group. Locks in the
        contract that `_LEGACY_NAMES` keeps them together.
        """
        config_dir = tmp_path / ".deepagents"
        state_dir = config_dir / ".state"
        config_dir.mkdir()
        (config_dir / "sessions.db").write_bytes(b"main")
        (config_dir / "sessions.db-wal").write_bytes(b"wal")
        (config_dir / "sessions.db-shm").write_bytes(b"shm")

        migrate_legacy_state(config_dir=config_dir, state_dir=state_dir)

        assert (state_dir / "sessions.db").read_bytes() == b"main"
        assert (state_dir / "sessions.db-wal").read_bytes() == b"wal"
        assert (state_dir / "sessions.db-shm").read_bytes() == b"shm"
        for name in ("sessions.db", "sessions.db-wal", "sessions.db-shm"):
            assert not (config_dir / name).exists(), name

    def test_state_dir_mkdir_failure_is_logged(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If `.state/` cannot be created, warn and leave legacy alone."""
        config_dir = tmp_path / ".deepagents"
        config_dir.mkdir()
        # `.state` is a regular file — `mkdir(exist_ok=True)` will raise.
        state_path = config_dir / ".state"
        state_path.write_text("not a directory")
        (config_dir / "sessions.db").write_bytes(b"db")

        with caplog.at_level("WARNING", logger="deepagents_cli.state_migration"):
            migrate_legacy_state(config_dir=config_dir, state_dir=state_path)

        # Legacy untouched, warning emitted.
        assert (config_dir / "sessions.db").read_bytes() == b"db"
        assert state_path.read_text() == "not a directory"
        assert any(
            "Could not create state directory" in record.getMessage()
            for record in caplog.records
        )

    def test_per_entry_rename_failure_does_not_block_others(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed rename on one entry must not abort the rest.

        Mirrors the partial-failure path callers rely on (e.g., one
        sidecar locked on Windows shouldn't strand `history.jsonl`).
        """
        config_dir = tmp_path / ".deepagents"
        state_dir = config_dir / ".state"
        config_dir.mkdir()
        (config_dir / "sessions.db").write_bytes(b"db")
        (config_dir / "history.jsonl").write_text("h\n")

        original_rename = Path.rename

        def selective_rename(self: Path, target: Path) -> Path:
            if self.name == "sessions.db":
                msg = "simulated lock"
                raise OSError(msg)
            return original_rename(self, target)

        monkeypatch.setattr(Path, "rename", selective_rename)

        with caplog.at_level("WARNING", logger="deepagents_cli.state_migration"):
            migrate_legacy_state(config_dir=config_dir, state_dir=state_dir)

        # Failing entry stays at legacy; non-failing entry moved.
        assert (config_dir / "sessions.db").read_bytes() == b"db"
        assert not (state_dir / "sessions.db").exists()
        assert (state_dir / "history.jsonl").read_text() == "h\n"
        assert not (config_dir / "history.jsonl").exists()
        # Failure is logged with file context.
        assert any(
            "Failed to migrate" in record.getMessage()
            and "sessions.db" in record.getMessage()
            for record in caplog.records
        )
