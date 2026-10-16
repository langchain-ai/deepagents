"""Server-side side-question cancellation without a network connection."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from starlette.requests import Request

from deepagents_code import btw_api, offload_api
from deepagents_code.btw import BtwOperation
from deepagents_code.btw_history import append_exchange, load_history

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from httpx import AsyncClient, Response
    from starlette.types import Message


@pytest.fixture(autouse=True)
def side_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Store side history in an isolated sessions database."""
    path = tmp_path / "sessions.db"
    monkeypatch.setenv("DEEPAGENTS_CODE_SERVER_DB_PATH", str(path))
    return path


async def test_client_sent_history_rejected_before_workspace_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from httpx import ASGITransport, AsyncClient

    workspace = AsyncMock()
    monkeypatch.setattr(btw_api, "require_thread_workspace", workspace)
    async with AsyncClient(
        transport=ASGITransport(app=offload_api.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/dcode/threads/thread/btw",
            json={"question": "why", "workspace": {}, "history": []},
        )
    assert response.status_code == 422
    workspace.assert_not_awaited()


@pytest.mark.parametrize(
    "outcome", ["disconnect", "cancel", "complete", "error", "timeout"]
)
@pytest.mark.parametrize("streaming", [False, True])
async def test_side_request_cleans_up_generation_and_disconnect_listener(
    outcome: str, streaming: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()
    listener_stopped = asyncio.Event()
    result: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    deadlines: list[asyncio.Timeout] = []
    if outcome == "timeout":
        timeout = asyncio.timeout

        def no_deadline(_seconds: float) -> asyncio.Timeout:
            deadline = timeout(None)
            deadlines.append(deadline)
            return deadline

        monkeypatch.setattr(btw_api.asyncio, "timeout", no_deadline)
    incoming: asyncio.Queue[Message] = asyncio.Queue()
    incoming.put_nowait(
        {
            "type": "http.request",
            "body": json.dumps({"question": "why", "workspace": {}}).encode(),
            "more_body": False,
        }
    )

    async def receive() -> Message:
        try:
            return await incoming.get()
        finally:
            if started.is_set():
                listener_stopped.set()

    async def answer(
        _thread: str,
        _state: object,
        _question: str,
        *,
        history: object = (),
        on_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        assert not history
        if on_text is not None:
            await on_text("side ")
        started.set()
        try:
            return await result
        finally:
            stopped.set()

    operation = BtwOperation(
        FakeMessagesListChatModel(responses=[AIMessage(content="unused")]), "", None
    )
    monkeypatch.setattr(operation, "answer", answer)
    monkeypatch.setattr(btw_api, "require_thread_workspace", AsyncMock())
    monkeypatch.setattr(
        offload_api,
        "get_server_runtime",
        AsyncMock(
            return_value=SimpleNamespace(backend=SimpleNamespace(_dcode_btw=operation))
        ),
    )
    monkeypatch.setattr(
        offload_api,
        "_thread_client",
        lambda: SimpleNamespace(
            threads=SimpleNamespace(get_state=AsyncMock(return_value={"values": {}}))
        ),
    )
    request = Request(
        {
            "type": "http",
            "headers": [(b"accept", b"text/event-stream")] if streaming else [],
            "path_params": {"thread_id": "thread"},
        },
        receive,
    )
    sent: list[Message] = []

    send = AsyncMock(side_effect=sent.append)

    async def run() -> None:
        response = await btw_api.btw(request)
        await response(request.scope, receive, send)

    handler = asyncio.create_task(run())
    try:
        await asyncio.wait_for(started.wait(), 2)
        if streaming:
            assert b'event: text\ndata: "side "\n\n' in sent[1]["body"]
            assert not handler.done()
        if outcome == "disconnect":
            incoming.put_nowait({"type": "http.disconnect"})
        elif outcome == "cancel":
            handler.cancel()
        elif outcome == "error":
            result.set_exception(ValueError("provider failed"))
        elif outcome == "timeout":
            deadlines[-1].reschedule(asyncio.get_running_loop().time())
        else:
            result.set_result("side answer")

        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await handler
        else:
            # Shield so a test timeout cannot itself cancel generation and mask the bug.
            await asyncio.wait_for(asyncio.shield(handler), 2)
            body = b"".join(message.get("body", b"") for message in sent)
            if streaming:
                assert sent[0]["status"] == 200
                if outcome == "complete":
                    assert b'event: complete\ndata: {"text": "side answer"' in body
                elif outcome in {"error", "timeout"}:
                    assert b"event: error\n" in body
                    assert b"provider failed" not in body
            else:
                assert (
                    sent[0]["status"]
                    == {
                        "disconnect": 499,
                        "complete": 200,
                        "error": 500,
                        "timeout": 504,
                    }[outcome]
                )
                if outcome == "complete":
                    assert json.loads(body) == {"text": "side answer"}
        # Only an answer that completed while connected joins the side history.
        expected = [("why", "side answer")] if outcome == "complete" else []
        assert load_history("thread") == (expected, 0)
        assert stopped.is_set()
        assert listener_stopped.is_set()
    finally:
        handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)


@pytest.mark.parametrize(
    "selection",
    [{"model": "provider:model"}, {"model_params": {"temperature": 0.2}}],
)
async def test_selection_rejected_before_workspace_or_model_access(
    selection: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from httpx import ASGITransport, AsyncClient

    workspace = AsyncMock()
    monkeypatch.setattr(btw_api, "require_thread_workspace", workspace)
    async with AsyncClient(
        transport=ASGITransport(app=offload_api.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/dcode/threads/thread/btw",
            json={"question": "why", "workspace": {}, **selection},
        )
    assert response.status_code == 422
    workspace.assert_not_awaited()


class _SideServer:
    """Route requests to a scripted side-answer generator."""

    def __init__(self, client: AsyncClient) -> None:
        self.client = client
        self.histories: list[list[tuple[str, str]]] = []
        self.replies: dict[str, asyncio.Future[str]] = {}
        self.started = asyncio.Semaphore(0)

        async def answer(
            _thread: str,
            _state: object,
            question: str,
            *,
            history: list[tuple[str, str]],
            on_text: object = None,
        ) -> str:
            del on_text
            self.histories.append(list(history))
            reply = self.replies.pop(question, None)
            if reply is None:
                return f"answer to {question}"
            self.started.release()
            return await reply

        self.answer = answer

    async def ask(self, question: str, *, streaming: bool = False) -> Response:
        headers = {"accept": "text/event-stream"} if streaming else {}
        return await self.client.post(
            "/dcode/threads/thread/btw",
            json={"question": question, "workspace": {}},
            headers=headers,
        )

    async def clear(self) -> Response:
        return await self.client.post(
            "/dcode/threads/thread/btw/clear", json={"workspace": {}}
        )

    async def history(self) -> list[list[str]]:
        response = await self.client.post(
            "/dcode/threads/thread/btw/history", json={"workspace": {}}
        )
        response.raise_for_status()
        return response.json()["history"]


@pytest.fixture
async def side_server(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_SideServer]:
    from httpx import ASGITransport, AsyncClient

    operation = BtwOperation(
        FakeMessagesListChatModel(responses=[AIMessage(content="unused")]), "", None
    )
    monkeypatch.setattr(btw_api, "require_thread_workspace", AsyncMock())
    monkeypatch.setattr(
        offload_api,
        "get_server_runtime",
        AsyncMock(
            return_value=SimpleNamespace(backend=SimpleNamespace(_dcode_btw=operation))
        ),
    )
    monkeypatch.setattr(
        offload_api,
        "_thread_client",
        lambda: SimpleNamespace(
            threads=SimpleNamespace(get_state=AsyncMock(return_value={"values": {}}))
        ),
    )
    async with AsyncClient(
        transport=ASGITransport(app=offload_api.app), base_url="http://test"
    ) as client:
        server = _SideServer(client)
        monkeypatch.setattr(operation, "answer", server.answer)
        yield server


@pytest.mark.parametrize("streaming", [False, True])
async def test_side_history_persists_across_requests(
    side_server: _SideServer, streaming: bool
) -> None:
    assert (await side_server.ask("  First  ", streaming=streaming)).is_success
    assert (await side_server.ask("Second", streaming=streaming)).is_success
    assert side_server.histories == [[], [("First", "answer to First")]]
    assert await side_server.history() == [
        ["First", "answer to First"],
        ["Second", "answer to Second"],
    ]


async def test_failed_answer_is_not_stored(side_server: _SideServer) -> None:
    reply: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    reply.set_exception(ValueError("provider failed"))
    side_server.replies["why"] = reply
    assert (await side_server.ask("why")).status_code == 500
    assert await side_server.history() == []


@pytest.mark.parametrize("streaming", [False, True])
async def test_clear_discards_answer_in_progress(
    side_server: _SideServer, streaming: bool
) -> None:
    await side_server.ask("Kept until clear")
    reply: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    side_server.replies["In flight"] = reply
    pending = asyncio.create_task(side_server.ask("In flight", streaming=streaming))
    await asyncio.wait_for(side_server.started.acquire(), 2)
    cleared = await side_server.clear()
    assert cleared.json() == {"history": []}
    reply.set_result("late answer")
    response = await asyncio.wait_for(pending, 2)
    assert "late answer" in response.text
    assert await side_server.history() == []
    await side_server.ask("Fresh")
    assert side_server.histories[-1] == []
    assert await side_server.history() == [["Fresh", "answer to Fresh"]]


async def test_overlapping_answers_store_every_exchange_whole(
    side_server: _SideServer,
) -> None:
    replies = [asyncio.get_running_loop().create_future() for _ in range(2)]
    side_server.replies.update(zip(("One", "Two"), replies, strict=True))
    first = asyncio.create_task(side_server.ask("One"))
    second = asyncio.create_task(side_server.ask("Two"))
    for _ in replies:
        await asyncio.wait_for(side_server.started.acquire(), 2)
    replies[1].set_result("two")
    await second
    replies[0].set_result("one")
    await first
    assert await side_server.history() == [["Two", "two"], ["One", "one"]]


async def test_stored_history_over_cap_is_rejected(side_server: _SideServer) -> None:
    append_exchange("thread", 0, "q", "x" * 128_000)
    response = await side_server.ask("why")
    assert response.status_code == 422
    assert "Ctrl+X" in response.json()["detail"]
    assert side_server.histories == []
    assert (await side_server.clear()).is_success
    assert (await side_server.ask("why")).is_success


async def test_history_access_requires_workspace_binding(
    side_server: _SideServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.workspace import WorkspaceConflictError

    await side_server.ask("Keep")
    monkeypatch.setattr(
        btw_api,
        "require_thread_workspace",
        AsyncMock(side_effect=WorkspaceConflictError("wrong workspace")),
    )
    assert (await side_server.clear()).status_code == 409
    history = await side_server.client.post(
        "/dcode/threads/thread/btw/history", json={"workspace": {}}
    )
    assert history.status_code == 409
    response = await side_server.client.post("/dcode/threads/thread/btw/clear", json={})
    assert response.status_code == 422
    monkeypatch.setattr(btw_api, "require_thread_workspace", AsyncMock())
    assert await side_server.history() == [["Keep", "answer to Keep"]]


async def test_thread_deletion_erases_side_history(
    side_server: _SideServer, side_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code import sessions

    monkeypatch.setattr(sessions, "get_db_path", lambda: side_db)
    await side_server.ask("Secret")
    stale = load_history("thread")[1]
    await sessions.delete_thread("thread")
    assert await side_server.history() == []
    # An answer that started before deletion cannot restore the history.
    assert not append_exchange("thread", stale, "Late", "answer")
    assert await side_server.history() == []
