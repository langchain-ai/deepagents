from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage, ChannelStatus
from deepagents_talon.observability import langsmith_tracing_enabled, log_event

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator


class RecordingAgent:
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def invoke(self, request: AgentRequest) -> AgentResult:
        return AgentResult(text=f"reply:{request.text}")


class RecordingChannel:
    def __init__(self) -> None:
        self.handler: Callable[[ChannelMessage], Awaitable[None]] | None = None
        self.sent: list[tuple[str, str]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def set_message_handler(self, handler: Callable[[ChannelMessage], Awaitable[None]]) -> None:
        self.handler = handler

    async def send_message(self, conversation_id: str, text: str) -> None:
        self.sent.append((conversation_id, text))

    async def send_media(self, conversation_id: str, media: object) -> None:
        pass

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> None:
        pass

    async def status(self) -> ChannelStatus:
        return ChannelStatus(provider="test", connected=True)


def test_langsmith_tracing_requires_opt_in_and_api_key() -> None:
    assert langsmith_tracing_enabled({"LANGSMITH_TRACING": "true"}) is False
    assert langsmith_tracing_enabled({"LANGSMITH_API_KEY": "key"}) is False
    assert (
        langsmith_tracing_enabled({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "key"}) is True
    )


async def test_host_wraps_agent_run_in_langsmith_context(tmp_path, monkeypatch) -> None:
    contexts: list[dict[str, object]] = []

    @contextmanager
    def tracing_context(**kwargs: object) -> Iterator[None]:
        contexts.append(kwargs)
        yield

    monkeypatch.setattr("langsmith.tracing_context", tracing_context)
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "key",
            "LANGSMITH_PROJECT": "talon-tests",
        },
        base_home=tmp_path,
    )
    channel = RecordingChannel()
    host = TalonHost(config=config, agent=RecordingAgent(), channels=[channel])

    await host.start()
    await host.receive_message(
        channel,
        ChannelMessage(conversation_id="chat", text="hello", sender_id="sender"),
    )
    await _wait_for_sent_count(channel, 1)
    await host.stop()

    assert channel.sent == [("chat", "reply:hello")]
    assert contexts == [
        {
            "project_name": "talon-tests",
            "tags": ["deepagents-talon", "assistant:assistant"],
            "metadata": {
                "assistant_id": "assistant",
                "channel": "test",
                "conversation_id": "chat",
                "sender_id": "sender",
                "message_id": None,
            },
            "enabled": True,
        },
    ]


def test_log_event_emits_json_payload(caplog) -> None:
    logger = logging.getLogger("deepagents_talon.tests")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_event(logger, "cron.tick", due_count=2)

    payload = caplog.messages[0].removeprefix("talon_event ")
    assert json.loads(payload) == {"event": "cron.tick", "due_count": 2}


def test_log_event_redacts_secrets_and_url_credentials(caplog) -> None:
    logger = logging.getLogger("deepagents_talon.tests")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_event(
            logger,
            "secret.check",
            conversation_id="chat-123",
            endpoint="https://user:pass@example.com/mcp?api_key=secret-token",
            headers={"Authorization": "Bearer raw-token"},
        )

    payload = json.loads(caplog.messages[0].removeprefix("talon_event "))
    assert payload == {
        "conversation_id": "[redacted]",
        "endpoint": "https://example.com/mcp",
        "event": "secret.check",
        "headers": {"Authorization": "[redacted]"},
    }
    assert "secret-token" not in caplog.text
    assert "raw-token" not in caplog.text
    assert "chat-123" not in caplog.text


async def _wait_for_sent_count(channel: RecordingChannel, count: int) -> None:
    for _ in range(100):
        if len(channel.sent) >= count:
            return
        await asyncio.sleep(0)
    msg = f"channel sent {len(channel.sent)} message(s), expected {count}"
    raise AssertionError(msg)
