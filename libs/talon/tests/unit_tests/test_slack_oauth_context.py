"""Historical OAuth callbacks must never reenter model input through Slack."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from deepagents_talon.channels.slack import _SlackSdkGateway
from tests.archive_helpers import make_runtime, make_saver
from tests.integration_tests.test_slack_host import _drain, _host, _mention
from tests.unit_tests.test_tool_approval_runtime import ToolModel

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.messages import BaseMessage

CALLBACK = "http://localhost:3000/callback?code=TEST_SECRET&state=TEST_STATE"
HISTORICAL_CALLBACKS = [
    CALLBACK,
    f"<{CALLBACK.replace('&', '&amp;')}|finish login>",
    f"<@UBOT> {CALLBACK}",
    f"Here is the callback ({CALLBACK}). Thanks!",
    CALLBACK.replace("localhost:3000", "127.0.0.1:6359"),
    CALLBACK.replace("localhost:3000", "[::1]:6359"),
    CALLBACK.replace("localhost:3000", "localhost"),
    CALLBACK.replace("code=TEST_SECRET", "error=TEST_SECRET"),
    "x" * 950 + " " + CALLBACK,
    CALLBACK.replace("&state=", "x" * 1000 + "&state="),
    "x" * 1100 + " " + CALLBACK,
]
ORDINARY = "see https://example.com/callback?code=public&state=public"


@pytest.mark.parametrize("callback", HISTORICAL_CALLBACKS)
async def test_gateway_excludes_callbacks_before_truncation(callback: str) -> None:
    gateway = _SlackSdkGateway(bot_token="b", app_token="a", timeout_seconds=1)  # noqa: S106  # inert test tokens
    gateway._web = AsyncMock()
    gateway._web.conversations_replies.return_value = {
        "messages": [
            {"ts": "1", "user": "UOP", "text": ORDINARY},
            {"ts": "2", "user": "UOP", "text": callback},
            {"ts": "3", "user": "UOP", "text": "neighbor"},
        ],
    }
    assert await gateway.thread_context("C1", "1", "4") == [
        ("UOP", ORDINARY),
        ("UOP", "neighbor"),
    ]


@pytest.mark.parametrize("callback", HISTORICAL_CALLBACKS)
async def test_host_excludes_retrieved_callbacks_without_pending_login(
    tmp_path: Path, callback: str
) -> None:
    host, gateway = _host(tmp_path)
    await host.start()
    try:
        await gateway.handle_message(_mention("1700000000.000100", text=CALLBACK))
        await _drain()
        assert host.agent.requests == []
        assert not host._pending_authorizations
        gateway.context = [("UOP", callback), ("UOTHER", "untrusted"), ("UOP", ORDINARY)]
        await gateway.handle_message(
            _mention("1700000000.000200", thread_ts="1700000000.000100", text="continue")
        )
        await _drain()
        assert len(host.agent.requests) == 1
        request = host.agent.requests[0]
        assert request.metadata["slack_thread_context"] == f"UOP: {ORDINARY}"
        assert ORDINARY in request.text
        assert "untrusted" not in request.text
        assert "TEST_SECRET" not in repr(request)
        assert "TEST_STATE" not in repr(request)
    finally:
        await host.stop()


class RecordingModelCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, object],
        messages: list[list[BaseMessage]],
        **kwargs: object,
    ) -> None:
        del serialized, kwargs
        self.messages.extend(message for batch in messages for message in batch)


async def test_runtime_does_not_persist_or_expose_retrieved_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = RecordingModelCallback()
    model = ToolModel(responses=[AIMessage(content="done")], callbacks=[capture])
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    async with make_saver(tmp_path / "history.sqlite") as saver:
        host, gateway = _host(tmp_path)
        runtime = make_runtime(saver, tmp_path)
        host.agent = runtime
        await host.start()
        try:
            await gateway.handle_message(_mention("1700000000.000100", text=CALLBACK))
            assert not capture.messages
            gateway.context = [("UOP", text) for text in [*HISTORICAL_CALLBACKS, ORDINARY]]
            await gateway.handle_message(
                _mention("1700000000.000200", thread_ts="1700000000.000100", text="continue")
            )
            async with asyncio.timeout(8):
                await asyncio.gather(*host._tasks.values())
            assert any(text == "done" for _, text, _ in gateway.posts)
            scope = {"talon_history_channel": "slack", "talon_history_chat": "C1"}
            entries = await saver.archive.entries(scope)
            assert entries
            assert capture.messages
            assert ORDINARY in repr(entries)
            assert ORDINARY in repr(capture.messages)
            sessions = await saver.archive.sessions(scope)
            checkpoints = [
                checkpoint
                for session in sessions
                async for checkpoint in saver.alist({"configurable": {"thread_id": session}})
            ]
            assert checkpoints
            for content in (repr(entries), repr(capture.messages), repr(checkpoints)):
                assert "TEST_SECRET" not in content
                assert "TEST_STATE" not in content
        finally:
            await host.stop()
