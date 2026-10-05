"""Manual thread naming and active-name synchronization."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Static

from deepagents_code.app import DeepAgentsApp

if TYPE_CHECKING:
    from deepagents_code.sessions import ThreadInfo


@pytest.fixture
def naming_app(monkeypatch: pytest.MonkeyPatch) -> DeepAgentsApp:
    app = DeepAgentsApp(thread_id="original")
    monkeypatch.setattr(app, "notify", MagicMock())
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    monkeypatch.setattr(
        "deepagents_code.sessions.get_thread_name", AsyncMock(return_value=None)
    )
    return app


@pytest.mark.parametrize("manual_name", [False, True])
async def test_stale_load_cannot_overwrite_current_name(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch, *, manual_name: bool
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def get_name(_thread_id: str) -> str:
        started.set()
        await release.wait()
        return "Old name"

    monkeypatch.setattr("deepagents_code.sessions.get_thread_name", get_name)
    monkeypatch.setattr(
        "deepagents_code.sessions.rename_thread", AsyncMock(return_value=True)
    )
    task = asyncio.create_task(naming_app._load_thread_name())
    await started.wait()
    if manual_name:
        await naming_app._handle_command("/rename Current name")
    else:
        naming_app._lc_thread_id = "new"
        naming_app._thread_name = "Current name"
    release.set()
    await task
    assert naming_app._thread_name == "Current name"


async def test_rename_refreshes_open_thread_selector(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.tui.widgets.thread_selector import ThreadSelectorScreen

    thread: ThreadInfo = {
        "thread_id": "original",
        "agent_name": "agent",
        "updated_at": None,
        "latest_checkpoint_id": "cp_1",
        "thread_name": "Old name",
    }
    monkeypatch.setattr(
        "deepagents_code.sessions.list_threads",
        AsyncMock(side_effect=lambda **_: [thread.copy()]),
    )
    monkeypatch.setattr(
        ThreadSelectorScreen, "_load_available_agent_names", AsyncMock()
    )

    async def rename(*_args: object, **_kwargs: object) -> bool:
        await asyncio.sleep(0)
        thread["thread_name"] = "Cache repair"
        return True

    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    async with naming_app.run_test() as pilot:
        selector = ThreadSelectorScreen(
            initial_threads=[thread.copy()], filter_cwd=None
        )
        naming_app.push_screen(selector)
        await pilot.pause()
        name_cell = "ThreadOption .thread-cell-thread_name"
        assert str(selector.query_one(name_cell, Static).render()) == (
            thread["thread_name"] or ""
        )
        await naming_app._handle_command("/rename Cache repair")
        await pilot.pause()
        assert str(selector.query_one(name_cell, Static).render()) == "Cache repair"
