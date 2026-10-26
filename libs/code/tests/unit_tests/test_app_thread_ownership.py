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


@pytest.mark.parametrize("click", [False, True])
async def test_picker_conflict_keeps_filter_and_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, click: bool
) -> None:
    from textual.widgets import Input

    from deepagents_code.thread_ownership import ThreadOwnershipError
    from deepagents_code.tui.widgets.messages import AppMessage
    from deepagents_code.tui.widgets.thread_selector import (
        ThreadOption,
        ThreadSelectorScreen,
    )

    monkeypatch.setattr(sessions, "get_db_path", lambda: tmp_path / "sessions.db")
    threads: list[sessions.ThreadInfo] = [
        {
            "thread_id": "target",
            "initial_prompt": "Saved conversation",
            "agent_name": "agent",
            "updated_at": "2026-10-06T12:00:00+00:00",
            "cwd": str(tmp_path),
        }
    ]
    monkeypatch.setattr(sessions, "get_cached_threads", lambda **_kwargs: threads)
    monkeypatch.setattr(sessions, "list_threads", AsyncMock(return_value=threads))
    error = "Thread target is open elsewhere. Close it there to resume."
    reserve = MagicMock(side_effect=ThreadOwnershipError(error))
    app = DeepAgentsApp(thread_id="current", cwd=tmp_path)
    monkeypatch.setattr(app, "_reserve_thread", reserve)
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    load = AsyncMock()
    monkeypatch.setattr(app, "_resume_owned_thread", load)
    async with app.run_test(size=(120, 36)) as pilot:
        await app._handle_command("/threads")
        await pilot.pause()
        selector = app.screen
        assert isinstance(selector, ThreadSelectorScreen)
        await pilot.press("s", "a", "v", "e", "d")
        await pilot.pause()
        messages = list(app.query(AppMessage))
        if click:
            await pilot.click(ThreadOption)
        else:
            await pilot.press("enter")
        await pilot.pause()

        assert app.screen is selector
        assert selector.query_one("#thread-filter", Input).value == "saved"
        assert selector._selected_index == 0
        assert any(
            n.message == error and n.severity == "error" for n in app._notifications
        )
        assert list(app.query(AppMessage)) == messages
        assert app._lc_thread_id == "current"
        load.assert_not_awaited()
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not selector
        assert app._chat_input is not None
        assert app.focused is app._chat_input.input_widget
