"""Tests for `AnthropicCacheKeepaliveMiddleware`."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.middleware._cache_keepalive import AnthropicCacheKeepaliveMiddleware

INTERVAL = 0.05


def _request(*, anthropic: bool = True) -> ModelRequest:
    model: Any = (
        ChatAnthropic(model_name="claude-opus-5-5", anthropic_api_key="fake")  # type: ignore[call-arg]
        if anthropic
        else GenericFakeChatModel(messages=iter([]))
    )
    return ModelRequest(model=model, messages=[HumanMessage("hi")], state={"messages": []}, model_settings={"cache_control": {"type": "ephemeral"}})


def _response(*ids: str) -> ModelResponse:
    calls = [{"id": i, "name": "slow", "args": {}} for i in ids]
    return ModelResponse(result=[AIMessage(content="", tool_calls=calls)])


def _tool_request(tool_call_id: str) -> ToolCallRequest:
    return ToolCallRequest(tool_call={"id": tool_call_id, "name": "slow", "args": {}}, tool=None, state={}, runtime=None)  # type: ignore[arg-type]


class _Recorder:
    """Model handler that records ping requests after the first (real) call."""

    def __init__(self, *, fail_pings: bool = False) -> None:
        self.calls: list[ModelRequest] = []
        self.fail_pings = fail_pings

    def respond(self, request: ModelRequest, ids: tuple[str, ...]) -> ModelResponse:
        self.calls.append(request)
        if len(self.calls) > 1 and self.fail_pings:
            msg = "ping rejected"
            raise RuntimeError(msg)
        return _response(*ids) if len(self.calls) == 1 else ModelResponse(result=[AIMessage(content="")])

    @property
    def pings(self) -> list[ModelRequest]:
        return self.calls[1:]


async def _arun(mw: AnthropicCacheKeepaliveMiddleware, recorder: _Recorder, *, tool_seconds: float, anthropic: bool = True) -> ToolMessage:
    async def model_handler(request: ModelRequest) -> ModelResponse:
        return recorder.respond(request, ("t1",))

    async def tool_handler(request: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(tool_seconds)
        return ToolMessage("done", tool_call_id=request.tool_call["id"])

    await mw.awrap_model_call(_request(anthropic=anthropic), model_handler)
    result = await mw.awrap_tool_call(_tool_request("t1"), tool_handler)
    assert isinstance(result, ToolMessage)
    return result


async def test_pings_while_tool_runs_and_stops_after() -> None:
    mw = AnthropicCacheKeepaliveMiddleware(interval=INTERVAL, max_pings=50)
    recorder = _Recorder()
    await _arun(mw, recorder, tool_seconds=INTERVAL * 3.5)
    pings = len(recorder.pings)
    assert 2 <= pings <= 4
    await asyncio.sleep(INTERVAL * 3)
    assert len(recorder.pings) == pings


async def test_ping_resends_same_request_with_no_output() -> None:
    mw = AnthropicCacheKeepaliveMiddleware(interval=INTERVAL)
    recorder = _Recorder()
    await _arun(mw, recorder, tool_seconds=INTERVAL * 1.5)
    original, ping = recorder.calls[0], recorder.pings[0]
    assert ping.messages == original.messages
    assert ping.model_settings == {**original.model_settings, "max_tokens": 0, "stream": False}


async def test_short_tool_sends_no_ping() -> None:
    mw = AnthropicCacheKeepaliveMiddleware(interval=INTERVAL)
    recorder = _Recorder()
    await _arun(mw, recorder, tool_seconds=INTERVAL / 5)
    assert recorder.pings == []


async def test_hung_tool_is_capped_at_max_pings() -> None:
    mw = AnthropicCacheKeepaliveMiddleware(interval=INTERVAL, max_pings=2)
    recorder = _Recorder()
    await _arun(mw, recorder, tool_seconds=INTERVAL * 6)
    assert len(recorder.pings) == 2


async def test_failed_ping_stops_pinging_and_tool_still_returns() -> None:
    mw = AnthropicCacheKeepaliveMiddleware(interval=INTERVAL)
    recorder = _Recorder(fail_pings=True)
    result = await _arun(mw, recorder, tool_seconds=INTERVAL * 4)
    assert result.content == "done"
    assert len(recorder.pings) == 1


async def test_non_anthropic_model_is_not_pinged() -> None:
    mw = AnthropicCacheKeepaliveMiddleware(interval=INTERVAL)
    recorder = _Recorder()
    await _arun(mw, recorder, tool_seconds=INTERVAL * 3, anthropic=False)
    assert recorder.pings == []


async def test_parallel_tool_calls_share_one_ping_loop() -> None:
    mw = AnthropicCacheKeepaliveMiddleware(interval=INTERVAL, max_pings=50)
    recorder = _Recorder()

    async def model_handler(request: ModelRequest) -> ModelResponse:
        return recorder.respond(request, ("a", "b"))

    async def tool_handler(request: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(INTERVAL * (3.5 if request.tool_call["id"] == "b" else 0.5))
        return ToolMessage("done", tool_call_id=request.tool_call["id"])

    await mw.awrap_model_call(_request(), model_handler)
    await asyncio.gather(mw.awrap_tool_call(_tool_request("a"), tool_handler), mw.awrap_tool_call(_tool_request("b"), tool_handler))
    assert 2 <= len(recorder.pings) <= 4


def test_sync_path_pings_and_stops() -> None:
    mw = AnthropicCacheKeepaliveMiddleware(interval=INTERVAL, max_pings=50)
    recorder = _Recorder()
    lock = threading.Lock()

    def model_handler(request: ModelRequest) -> ModelResponse:
        with lock:
            return recorder.respond(request, ("t1",))

    def tool_handler(request: ToolCallRequest) -> ToolMessage:
        time.sleep(INTERVAL * 3.5)
        return ToolMessage("done", tool_call_id=request.tool_call["id"])

    mw.wrap_model_call(_request(), model_handler)
    mw.wrap_tool_call(_tool_request("t1"), tool_handler)
    pings = len(recorder.pings)
    assert 2 <= pings <= 4
    time.sleep(INTERVAL * 3)
    assert len(recorder.pings) == pings


def test_unknown_tool_call_passes_through() -> None:
    mw = AnthropicCacheKeepaliveMiddleware(interval=INTERVAL)
    result = mw.wrap_tool_call(_tool_request("missing"), lambda r: ToolMessage("ok", tool_call_id=r.tool_call["id"]))
    assert isinstance(result, ToolMessage)
    assert result.content == "ok"
