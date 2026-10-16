"""Groq tool-limit failures remain actionable across the server boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models import BaseChatModel
from langgraph.pregel.remote import RemoteException, RemoteGraph
from langgraph_api.serde import json_dumpb, json_loads
from langgraph_sdk.client import LangGraphClient

from deepagents_code.app import DeepAgentsApp
from deepagents_code.client.remote_client import RemoteAgent, format_agent_exception
from deepagents_code.model_retry import CodeModelRetryMiddleware, GroqToolLimitError
from deepagents_code.tui.widgets.messages import ErrorMessage

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatResult


_MESSAGE = (
    "Groq supports at most 128 tools per request. Disable some MCP tools "
    "or switch to a model/provider with a higher tool limit."
)
_PROVIDER_MESSAGE = "'tools' : maximum number of items is 128"
_THREAD_ID = "01966f3a-0000-7000-8000-000000000001"


class BadRequestError(Exception):
    __module__ = "groq"
    status_code = 400

    def __init__(
        self, message: str = _PROVIDER_MESSAGE, *, nested: bool = False
    ) -> None:
        super().__init__("provider text containing private request details")
        error = {"message": message, "type": "invalid_request_error"}
        self.body = {"error": error} if nested else error


class _RejectingModel(BaseChatModel):
    attempts: int = 0
    error: Exception

    @property
    def _llm_type(self) -> str:
        return "mock-groq"

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        self.attempts += 1
        raise self.error


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_tool_limit_is_safe_and_not_retried(
    *, nested: bool, asynchronous: bool
) -> None:
    middleware = CodeModelRetryMiddleware(max_retries=5)
    request = cast(
        "ModelRequest",
        SimpleNamespace(model=None, runtime=SimpleNamespace(stream_writer=None)),
    )
    handler = Mock(side_effect=BadRequestError(nested=nested))

    ahandler = AsyncMock(side_effect=handler)
    if asynchronous:
        with pytest.raises(GroqToolLimitError, match=r"^Groq supports") as caught:
            await middleware.awrap_model_call(request, ahandler)
    else:
        with pytest.raises(GroqToolLimitError, match=r"^Groq supports") as caught:
            middleware.wrap_model_call(request, handler)
    assert str(caught.value) == _MESSAGE
    assert handler.call_count == 1
    assert json_loads(json_dumpb(caught.value)) == {
        "error": "GroqToolLimitError",
        "message": _MESSAGE,
    }


@pytest.mark.parametrize(
    "error",
    [
        BadRequestError("invalid tool schema"),
        BadRequestError("maximum context length exceeded"),
        BadRequestError("'tools' : maximum number of items is 129"),
        ContextOverflowError("maximum context length exceeded"),
        httpx.HTTPStatusError(
            _PROVIDER_MESSAGE,
            request=httpx.Request("POST", "https://example.com"),
            response=httpx.Response(400),
        ),
    ],
)
async def test_unrelated_errors_are_preserved(error: Exception) -> None:
    middleware = CodeModelRetryMiddleware(max_retries=5)
    request = cast(
        "ModelRequest",
        SimpleNamespace(model=None, runtime=SimpleNamespace(stream_writer=None)),
    )
    handler = Mock(side_effect=error)

    ahandler = AsyncMock(side_effect=handler)

    with pytest.raises(type(error)) as caught:
        await middleware.awrap_model_call(request, ahandler)
    assert caught.value is error
    assert handler.call_count == 1
    assert (
        json_loads(json_dumpb(caught.value))["message"] == "An internal error occurred"
    )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("status_code", 429),
        ("body", None),
        ("body", {"message": _PROVIDER_MESSAGE, "type": "other_error"}),
        ("__module__", "openai"),
    ],
)
def test_detection_requires_groq_status_and_structured_body(
    monkeypatch: pytest.MonkeyPatch, attribute: str, value: object
) -> None:
    from deepagents_code.model_retry import _is_groq_tool_limit_error

    error = BadRequestError()
    target = BadRequestError if attribute == "__module__" else error
    monkeypatch.setattr(target, attribute, value)
    assert not _is_groq_tool_limit_error(error)


async def test_provider_rejection_streams_to_tui_as_error() -> None:
    model = _RejectingModel(error=BadRequestError(nested=True))
    graph = create_agent(model, middleware=[CodeModelRetryMiddleware(max_retries=5)])
    frames: list[bytes] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/threads/{_THREAD_ID}/runs/stream"
        try:
            async for namespace, mode, data in graph.astream(
                {
                    "messages": [
                        {"role": "user", "content": "tell me a 5 paragraph story"}
                    ]
                },
                stream_mode=["messages", "updates", "custom"],
                subgraphs=True,
            ):
                event = "|".join((mode, *namespace))
                frames.append(
                    b"event: "
                    + event.encode()
                    + b"\ndata: "
                    + json_dumpb(data)
                    + b"\n\n"
                )
        except (GroqToolLimitError, BadRequestError) as exc:
            frames.append(b"event: error\ndata: " + json_dumpb(exc) + b"\n\n")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(frames),
        )

    assert (
        json_loads(json_dumpb(model.error))["message"] == "An internal error occurred"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://test"
    ) as client:
        remote = RemoteAgent("http://test", local_ownership=False)
        remote._graph = RemoteGraph("agent", client=LangGraphClient(client))
        remote._workspaces[_THREAD_ID] = {"workspace_id": "test-workspace"}
        with pytest.raises(RemoteException) as caught:
            async for _ in remote.astream(
                {"messages": []},
                config={"configurable": {"thread_id": _THREAD_ID}},
                stream_mode=["messages", "updates", "custom"],
                subgraphs=True,
            ):
                pass
    assert format_agent_exception(caught.value) == f"GroqToolLimitError: {_MESSAGE}"
    app = DeepAgentsApp(agent=MagicMock())
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.tui.textual_adapter.execute_task_textual",
            new_callable=AsyncMock,
            side_effect=caught.value,
        ):
            await app._run_agent_task("tell me a 5 paragraph story")
            await pilot.pause()
        errors = app.query(ErrorMessage)
        assert any(_MESSAGE in str(widget._content) for widget in errors)
    assert model.attempts == 1
    terminal = json_loads(frames[-1].split(b"data: ", 1)[1].strip())
    assert terminal == {"error": "GroqToolLimitError", "message": _MESSAGE}
    assert not any(b"model_retry" in frame for frame in frames)
    assert not any(b"private request details" in frame for frame in frames)
