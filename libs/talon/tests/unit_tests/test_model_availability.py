from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import anthropic
import httpx
import openai
import pytest

from deepagents_talon.host import TalonHost, _is_model_unavailable
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config

if TYPE_CHECKING:
    from pathlib import Path


class UnavailableAgent(BlockingAgent):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        raise self.error


def _error(error_type, body: object) -> Exception:
    response = httpx.Response(404, request=httpx.Request("POST", "https://provider.example"))
    return error_type("sensitive provider detail", response=response, body=body)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_error(anthropic.NotFoundError, {"error": {"type": "not_found_error"}}), True),
        (
            _error(anthropic.BadRequestError, {"details": {"error_code": "model_not_available"}}),
            True,
        ),
        (_error(openai.NotFoundError, {"code": "other_resource_missing"}), False),
        (_error(anthropic.BadRequestError, {"details": []}), False),
    ],
)
def test_model_availability(error: Exception, *, expected: bool) -> None:
    assert _is_model_unavailable(error) is expected


def test_model_availability_from_context() -> None:
    wrapper = RuntimeError("wrapped failure")
    wrapper.__context__ = _error(openai.NotFoundError, {"code": "model_not_found"})
    assert _is_model_unavailable(wrapper)


async def test_turn_reports_model_availability(tmp_path: Path) -> None:
    error = RuntimeError("wrapped failure")
    error.__cause__ = _error(openai.NotFoundError, {"code": "model_not_found"})
    channel = RecordingChannel()
    host = TalonHost(config=_config(tmp_path), agent=UnavailableAgent(error), channels=[channel])
    await host.start()
    host._model_selections["__default__"] = "provider:selected-model"
    try:
        await host.receive_message(channel, ChannelMessage("chat", "hello"))
        await asyncio.wait_for(host._tasks["test:chat"], 2)
        [(_, text)] = channel.sent
        assert "provider:selected-model" in text
        assert "/model default" in text
        assert "choose another model" in text
        assert "sensitive provider detail" not in text
    finally:
        await host.stop()
