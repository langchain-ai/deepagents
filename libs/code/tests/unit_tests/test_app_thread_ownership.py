"""Ownership selection and rollback at the app boundary."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepagents_code import sessions
from deepagents_code.app import DeepAgentsApp
from deepagents_code.thread_ownership import release_all, try_acquire

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def owned_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[DeepAgentsApp]:
    """Provide an isolated app and database for ownership checks."""
    monkeypatch.setattr(sessions, "get_db_path", lambda: tmp_path / "sessions.db")
    app = DeepAgentsApp(thread_id="current", server_kwargs={})
    monkeypatch.setattr(app, "notify", MagicMock())
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    app._reserve_thread("current")
    yield app
    release_all()


def seed_history() -> None:
    """Store repeated checkpoints with distinct recency and agent identities."""
    with closing(sqlite3.connect(sessions.get_db_path())) as conn, conn:
        conn.execute(
            "CREATE TABLE checkpoints "
            "(thread_id TEXT, checkpoint_id TEXT, metadata TEXT)"
        )
        conn.executemany(
            "INSERT INTO checkpoints VALUES (?, ?, ?)",
            [
                ("older", "1", '{"agent_name":"worker"}'),
                ("newer", "2", '{"agent_name":"worker"}'),
                ("newer", "3", '{"agent_name":"worker"}'),
                ("other", "4", '{"agent_name":"other"}'),
            ],
        )


@pytest.mark.usefixtures("owned_app")
async def test_distinct_candidates_and_agent_filter() -> None:
    seed_history()
    assert await sessions.get_recent_thread_ids() == ["other", "newer", "older"]
    assert await sessions.get_recent_thread_ids("worker") == ["newer", "older"]
    assert await sessions.get_recent_thread_ids("missing") == []


async def test_bare_resume_skips_and_reserves(
    owned_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    notice = MagicMock()
    monkeypatch.setattr(owned_app, "notify", notice)
    seed_history()
    occupied = try_acquire("newer")
    assert occupied is not None
    assert await owned_app._claim_recent_thread("worker") == "older"
    assert try_acquire("older") is None
    assert "Skipped 1" in notice.call_args.args[0]


async def test_all_occupied_offers_new(
    owned_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_history()
    assert try_acquire("newer") is not None
    assert try_acquire("older") is not None
    owned_app._assistant_id = "worker"
    owned_app._resume_thread_intent = "__MOST_RECENT__"
    choice = AsyncMock(return_value="new")
    monkeypatch.setattr(owned_app, "_push_screen_result_future", choice)
    monkeypatch.setattr(owned_app, "_sync_status_connection", MagicMock())
    monkeypatch.setattr(
        owned_app, "_restore_startup_tip_after_resume_fallback", AsyncMock()
    )
    await owned_app._resolve_resume_thread()
    assert choice.await_count == 1
    assert owned_app._lc_thread_id not in {"newer", "older", "current"}
    assert not owned_app._initial_resume_requested


async def test_failed_switch_preserves_previous(
    owned_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = AsyncMock(side_effect=RuntimeError("history failed"))
    monkeypatch.setattr(owned_app, "_resume_owned_thread", failure)
    with pytest.raises(RuntimeError, match="history failed"):
        await owned_app._resume_thread("target")
    assert try_acquire("current") is None
    available = try_acquire("target")
    assert available is not None


async def test_successful_switch_releases_previous(
    owned_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    def switch(thread_id: str) -> None:
        owned_app._lc_thread_id = thread_id

    monkeypatch.setattr(
        owned_app, "_resume_owned_thread", AsyncMock(side_effect=switch)
    )
    await owned_app._resume_thread("target")
    assert try_acquire("target") is None
    assert try_acquire("current") is not None


async def test_explicit_conflict_does_not_load_history(
    owned_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.thread_ownership import ThreadOwnershipError

    failure = MagicMock(
        side_effect=ThreadOwnershipError("Thread target is open elsewhere")
    )
    monkeypatch.setattr(owned_app, "_reserve_thread", failure)
    load = AsyncMock()
    monkeypatch.setattr(owned_app, "_resume_owned_thread", load)
    await owned_app._resume_thread("target")
    load.assert_not_awaited()
    assert owned_app._lc_thread_id == "current"
    assert try_acquire("current") is None
