"""Server-side side-question cancellation without a network connection."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from starlette.requests import Request

from deepagents_code import btw_api, config, offload_api
from deepagents_code.btw import BtwOperation

if TYPE_CHECKING:
    from starlette.types import Message


@pytest.mark.parametrize(
    "outcome", ["disconnect", "cancel", "complete", "error", "timeout"]
)
async def test_side_request_cleans_up_generation_and_disconnect_listener(
    outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()
    listener_stopped = asyncio.Event()
    result: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    deadline = asyncio.timeout(None)
    if outcome == "timeout":
        monkeypatch.setattr(btw_api.asyncio, "timeout", lambda _seconds: deadline)
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
        _thread: str, _state: object, _question: str, **_selection: object
    ) -> str:
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
    request = Request({"type": "http", "path_params": {"thread_id": "thread"}}, receive)
    handler = asyncio.create_task(btw_api.btw(request))
    try:
        await asyncio.wait_for(started.wait(), 2)
        if outcome == "disconnect":
            incoming.put_nowait({"type": "http.disconnect"})
        elif outcome == "cancel":
            handler.cancel()
        elif outcome == "error":
            result.set_exception(ValueError("provider failed"))
        elif outcome == "timeout":
            deadline.reschedule(asyncio.get_running_loop().time())
        else:
            result.set_result("side answer")

        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await handler
        else:
            # Shield so a test timeout cannot itself cancel generation and mask the bug.
            response = await asyncio.wait_for(asyncio.shield(handler), 2)
            assert (
                response.status_code
                == {"disconnect": 499, "complete": 200, "error": 500, "timeout": 504}[
                    outcome
                ]
            )
            if outcome == "complete":
                assert json.loads(bytes(response.body)) == {"text": "side answer"}
        assert stopped.is_set()
        assert listener_stopped.is_set()
    finally:
        handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)


@pytest.mark.parametrize(
    ("model", "params"),
    [
        ("google_genai:model", {"thinking_level": "high"}),
        (
            "google_genai:model",
            {"thinking_config": {"thinking_level": "high", "include_thoughts": True}},
        ),
        ("anthropic:model", {"output_config": {"effort": "high"}}),
    ],
)
async def test_provider_generation_overrides_reach_side_answer(
    model: str, params: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from httpx import ASGITransport, AsyncClient

    chat = FakeMessagesListChatModel(responses=[AIMessage(content="side answer")])
    create_model = Mock(return_value=SimpleNamespace(model=chat))
    monkeypatch.setattr(config, "create_model", create_model)
    operation = BtwOperation(chat, "", None)
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
        response = await client.post(
            "/dcode/threads/thread/btw",
            json={
                "question": "why",
                "workspace": {},
                "model": model,
                "model_params": params,
            },
        )
    assert response.status_code == 200
    assert response.json() == {"text": "side answer"}
    create_model.assert_called_once_with(
        model, extra_kwargs=params, bind_preserved_thinking=False
    )


@pytest.mark.parametrize(
    "selection",
    [
        {"model": 42},
        {"model": " "},
        {"model_params": {"temperature": 0.2}},
        {"model": "provider:model", "model_params": []},
        *[
            {"model": "provider:model", "model_params": {key: value}}
            for key, value in [
                ("base_url", "http://untrusted"),
                ("api_key", "untrusted"),
                ("model_kwargs", {"base_url": "http://untrusted"}),
                ("extra_body", {"tools": []}),
                ("tools", []),
            ]
        ],
    ],
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
