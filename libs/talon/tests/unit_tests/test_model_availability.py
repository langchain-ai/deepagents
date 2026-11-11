from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import anthropic
import httpx
import openai
import pytest

from deepagents_talon.host import _AGENT_FAILURE_MESSAGE, TalonHost
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
    ("case", "generic_failure"),
    [
        (_error(anthropic.NotFoundError, {"error": {"type": "not_found_error"}}), False),
        (_error(anthropic.NotFoundError, {"type": "not_found_error"}), False),
        (
            _error(
                anthropic.BadRequestError,
                {"details": {"error_code": "model_not_available"}},
            ),
            False,
        ),
        (
            _error(
                anthropic.BadRequestError,
                {"error": {"details": {"error_code": "model_not_available"}}},
            ),
            False,
        ),
        (_error(openai.NotFoundError, {"code": "model_not_found"}), False),
        (_error(openai.NotFoundError, {"error": {"code": "model_not_found"}}), False),
        (_error(anthropic.BadRequestError, {"error": {"type": "invalid_request_error"}}), True),
        (_error(anthropic.NotFoundError, None), True),
        (_error(anthropic.BadRequestError, {"details": []}), True),
        (_error(anthropic.BadRequestError, {"error": "not_found_error"}), True),
        (_error(anthropic.NotFoundError, ["not_found_error"]), True),
        (_error(openai.NotFoundError, {"error": None}), True),
        (_error(openai.NotFoundError, {"code": "other_resource_missing"}), True),
        (_error(openai.BadRequestError, {"code": "model_not_found"}), True),
    ],
)
@pytest.mark.parametrize("chain", ["direct", "cause", "context"])
async def test_turn_reports_model_availability(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    case: Exception,
    *,
    generic_failure: bool,
    chain: str,
) -> None:
    error = case
    if chain != "direct":
        wrapper = RuntimeError("wrapped failure")
        if chain == "cause":
            wrapper.__cause__ = error
        else:
            wrapper.__context__ = error
        error = wrapper
    channel = RecordingChannel()
    host = TalonHost(config=_config(tmp_path), agent=UnavailableAgent(error), channels=[channel])
    await host.start()
    host._model_selections["__default__"] = "provider:selected-model"
    try:
        with caplog.at_level(logging.ERROR, logger="deepagents_talon.host"):
            await host.receive_message(channel, ChannelMessage("chat", "hello"))
            await asyncio.wait_for(host._tasks["test:chat"], 2)
        assert len(channel.sent) == 1
        text = channel.sent[0][1]
        if generic_failure:
            assert text == _AGENT_FAILURE_MESSAGE
        else:
            assert "provider:selected-model" in text
            assert "/model default" in text
            assert "choose another model" in text
        assert "sensitive provider detail" not in text
        assert "sensitive provider detail" in caplog.text
        assert any(record.exc_info for record in caplog.records)
    finally:
        await host.stop()
