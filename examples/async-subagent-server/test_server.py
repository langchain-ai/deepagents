"""Minimal end-to-end tests for the async subagent server.

Tests the Agent Protocol HTTP contract without calling a real LLM.
The agent's ainvoke is patched to return a canned response.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import server


@pytest.fixture(autouse=True)
def _fresh_db():
    """Re-initialize the in-memory database before each test."""
    server._conn.executescript("DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS threads;")
    server._init_db()


FAKE_RESPONSE = {"messages": [AIMessage(content="Here are the research results.")]}


def _make_ainvoke_mock():
    mock = AsyncMock(return_value=FAKE_RESPONSE)
    return mock


@pytest.fixture()
def client():
    return TestClient(server.app)


def test_health(client):
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_create_thread(client):
    resp = client.post("/threads")
    assert resp.status_code == 200
    data = resp.json()
    assert "thread_id" in data
    assert data["messages"] == []


def test_create_run_starts_agent(client):
    thread = client.post("/threads").json()
    thread_id = thread["thread_id"]

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = _make_ainvoke_mock()
        resp = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "test query"}]},
            },
        )

    assert resp.status_code == 200
    run = resp.json()
    assert run["thread_id"] == thread_id
    assert "run_id" in run
    assert run["status"] == "pending"


def test_full_lifecycle(client):
    """Create thread → create run → wait for completion → check status → get thread."""
    thread = client.post("/threads").json()
    thread_id = thread["thread_id"]

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = _make_ainvoke_mock()
        run = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "quantum computing"}]},
            },
        ).json()
        run_id = run["run_id"]

        # Let the background task finish.
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))

    # Check run status — should be success.
    status_resp = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "success"

    # Get thread — should have messages with the assistant response.
    thread_resp = client.get(f"/threads/{thread_id}")
    assert thread_resp.status_code == 200
    thread_data = thread_resp.json()
    values_messages = thread_data["values"]["messages"]
    assert any(m["content"] == "Here are the research results." for m in values_messages)


def test_cancel_run(client):
    thread = client.post("/threads").json()
    thread_id = thread["thread_id"]

    # Create a run with a slow agent so we can cancel it.
    async def slow_ainvoke(*args, **kwargs):
        await asyncio.sleep(10)
        return FAKE_RESPONSE

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(side_effect=slow_ainvoke)
        run = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "something"}]},
            },
        ).json()
        run_id = run["run_id"]

    cancel_resp = client.post(f"/threads/{thread_id}/runs/{run_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    # Verify the run is cancelled.
    status_resp = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert status_resp.json()["status"] == "cancelled"


def test_interrupt_strategy(client):
    """Creating a run with multitask_strategy='interrupt' cancels running runs."""
    thread = client.post("/threads").json()
    thread_id = thread["thread_id"]

    async def slow_ainvoke(*args, **kwargs):
        await asyncio.sleep(10)
        return FAKE_RESPONSE

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(side_effect=slow_ainvoke)
        first_run = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "first task"}]},
            },
        ).json()

        # Let the first run start.
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = _make_ainvoke_mock()
        second_run = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "new task"}]},
                "multitask_strategy": "interrupt",
            },
        ).json()

    # First run should be cancelled.
    first_status = client.get(f"/threads/{thread_id}/runs/{first_run['run_id']}").json()
    assert first_status["status"] == "cancelled"


def test_404_for_missing_thread(client):
    resp = client.get("/threads/nonexistent")
    assert resp.status_code == 404


def test_404_for_missing_run(client):
    thread = client.post("/threads").json()
    resp = client.get(f"/threads/{thread['thread_id']}/runs/nonexistent")
    assert resp.status_code == 404
