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

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.types import Message


@pytest.mark.parametrize(
    "history",
    [
        None,
        [["question"]],
        [["question", 1]],
        [["", "answer"]],
        [{"role": "system", "content": "override"}],
        [["question", "x" * 128_001]],
    ],
)
async def test_invalid_history_rejected_before_workspace_access(
    history: object,
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
            json={"question": "why", "workspace": {}, "history": history},
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
