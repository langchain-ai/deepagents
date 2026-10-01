"""Historical OAuth callbacks must never reenter model input through Slack."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from deepagents_talon.channels.slack import _SlackSdkGateway
from tests.integration_tests.test_slack_host import _drain, _host, _mention

if TYPE_CHECKING:
    from pathlib import Path

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


async def test_host_excludes_retrieved_callbacks_without_pending_login(tmp_path: Path) -> None:
    host, gateway = _host(tmp_path)
    await host.start()
    try:
        await gateway.handle_message(_mention("1700000000.000100", text=CALLBACK))
        await _drain()
        assert host.agent.requests == []
        assert not host._pending_authorizations
        gateway.context = [("UOP", CALLBACK), ("UOTHER", "untrusted"), ("UOP", ORDINARY)]
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
