"""Truncated model replies are classified and surfaced, never silently dropped."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel.remote import RemoteException, RemoteGraph
from langgraph_api.serde import json_dumpb, json_loads
from langgraph_sdk.client import LangGraphClient

from deepagents_code._glm_5p2_profile import _GlmTerminalStallRecovery
from deepagents_code.app import DeepAgentsApp
from deepagents_code.client.remote_client import RemoteAgent, format_agent_exception
from deepagents_code.model_retry import CodeModelRetryMiddleware
from deepagents_code.output_truncation import (
    EmptyTruncatedReplyMiddleware,
    ModelOutputTruncatedError,
    OutputTruncation,
    output_truncation,
)
from deepagents_code.tui.widgets.messages import ErrorMessage

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage

_THREAD_ID = "01966f3a-0000-7000-8000-000000000002"

# The shape observed from `openai:gpt-6.1-sol` on the Responses API.
_RESPONSES_INCOMPLETE = {
    "id": "resp_123",
    "status": "incomplete",
    "incomplete_details": {"reason": "max_output_tokens"},
    "model_name": "gpt-6.1-sol",
    "model_provider": "openai",
}


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        pytest.param(
            _RESPONSES_INCOMPLETE,
            OutputTruncation("max_output_tokens", "max_output_tokens"),
            id="openai-responses-max-output-tokens",
        ),
        pytest.param(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            OutputTruncation("content_filter", "content_filter"),
            id="openai-responses-content-filter",
        ),
        pytest.param(
            {"status": "completed", "model_provider": "openai"},
            None,
            id="openai-responses-completed",
        ),
        pytest.param(
            {"finish_reason": "length"},
            OutputTruncation("max_output_tokens", "length"),
            id="chat-completions-length",
        ),
        pytest.param(
            {"finish_reason": "content_filter"},
            OutputTruncation("content_filter", "content_filter"),
            id="chat-completions-content-filter",
        ),
        pytest.param({"finish_reason": "stop"}, None, id="chat-completions-stop"),
        pytest.param(
            {"finish_reason": "tool_calls"}, None, id="chat-completions-tool-calls"
        ),
        pytest.param(
            {"stop_reason": "max_tokens"},
            OutputTruncation("max_output_tokens", "max_tokens"),
            id="anthropic-max-tokens",
        ),
        pytest.param(
            {"stop_reason": "refusal"},
            OutputTruncation("content_filter", "refusal"),
            id="anthropic-refusal",
        ),
        pytest.param(
            {"stop_reason": "model_context_window_exceeded"},
            OutputTruncation(
                "context_window_exceeded", "model_context_window_exceeded"
            ),
            id="anthropic-context-window",
        ),
        pytest.param({"stop_reason": "end_turn"}, None, id="anthropic-end-turn"),
        pytest.param(
            {"finish_reason": "MAX_TOKENS"},
            OutputTruncation("max_output_tokens", "MAX_TOKENS"),
            id="google-max-tokens",
        ),
        pytest.param(
            {"finish_reason": "SAFETY"},
            OutputTruncation("content_filter", "SAFETY"),
            id="google-safety",
        ),
        pytest.param({"finish_reason": "STOP"}, None, id="google-stop"),
        pytest.param(
            {"stopReason": "max_tokens"},
            OutputTruncation("max_output_tokens", "max_tokens"),
            id="bedrock-max-tokens",
        ),
        pytest.param(
            {"stopReason": "guardrail_intervened"},
            OutputTruncation("content_filter", "guardrail_intervened"),
            id="bedrock-guardrail",
        ),
        pytest.param({"stopReason": "end_turn"}, None, id="bedrock-end-turn"),
        pytest.param({}, None, id="no-metadata"),
    ],
)
def test_classifies_provider_stop_reasons(
    metadata: dict[str, Any], expected: OutputTruncation | None
) -> None:
    message = AIMessage(content="", response_metadata=metadata)
    assert output_truncation(message) == expected


def _response(
    content: str | list[str | dict[str, Any]] = "",
    *,
    metadata: dict[str, Any] | None = None,
    with_tool_call: bool = False,
) -> ModelResponse[Any]:
    tool_calls = (
        [{"id": "call-1", "name": "ls", "args": {}, "type": "tool_call"}]
        if with_tool_call
        else []
    )
    message = AIMessage(
        content=content,
        tool_calls=tool_calls,
        response_metadata=_RESPONSES_INCOMPLETE if metadata is None else metadata,
    )
    return ModelResponse(result=[message])


def _request(tool_count: int = 0) -> ModelRequest:
    tools = [object()] * tool_count
    return cast(
        "ModelRequest", SimpleNamespace(model="openai:gpt-6.1-sol", tools=tools)
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "content",
    [
        pytest.param([], id="empty-blocks"),
        pytest.param("  \n", id="blank-text"),
        pytest.param(
            [{"type": "reasoning", "summary": [], "id": "rs_1"}], id="reasoning-only"
        ),
    ],
)
async def test_empty_truncated_reply_raises(
    content: str | list[str | dict[str, Any]], *, asynchronous: bool
) -> None:
    middleware = EmptyTruncatedReplyMiddleware()
    response = _response(content)

    async def call() -> None:
        if asynchronous:
            await middleware.awrap_model_call(
                _request(), AsyncMock(return_value=response)
            )
        else:
            middleware.wrap_model_call(_request(), MagicMock(return_value=response))

    with pytest.raises(ModelOutputTruncatedError) as caught:
        await call()

    assert caught.value.truncation.reason == "max_output_tokens"
    assert caught.value.model == "gpt-6.1-sol"
    assert str(caught.value).startswith(
        "Model 'gpt-6.1-sol' stopped before replying: it reached its output token "
        "limit (max_output_tokens)."
    )


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_response("Partial answer"), id="partial-text"),
        pytest.param(_response(with_tool_call=True), id="partial-tool-call"),
        pytest.param(_response(metadata={"finish_reason": "stop"}), id="normal-empty"),
        pytest.param(
            _response("Done.", metadata={"finish_reason": "stop"}), id="normal-reply"
        ),
    ],
)
def test_reply_with_output_or_normal_stop_passes_through(
    response: ModelResponse[Any],
) -> None:
    middleware = EmptyTruncatedReplyMiddleware()
    assert middleware.wrap_model_call(_request(), lambda _: response) is response


def test_zero_usage_points_at_tool_count_not_output_limit() -> None:
    """OpenAI returns an empty, zero-usage reply when sent too many tools."""
    message = AIMessage(
        content=[],
        response_metadata=_RESPONSES_INCOMPLETE,
        usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    middleware = EmptyTruncatedReplyMiddleware()

    with pytest.raises(ModelOutputTruncatedError) as caught:
        middleware.wrap_model_call(
            _request(tool_count=540),
            lambda _: ModelResponse(result=[message]),
        )

    text = str(caught.value)
    assert "The request had 540 tools" in text
    assert "reasoning effort" not in text
    assert copy.deepcopy(caught.value).tool_count == 540


def test_error_message_survives_server_serialization() -> None:
    error = ModelOutputTruncatedError(
        OutputTruncation("content_filter", "SAFETY"), "gemini-3-pro"
    )
    payload = json_loads(json_dumpb(error))
    assert payload == {"error": "ModelOutputTruncatedError", "message": str(error)}
    assert "the provider blocked the reply (SAFETY)" in payload["message"]
    clone = copy.deepcopy(error)
    assert (clone.truncation, clone.model, str(clone)) == (
        error.truncation,
        error.model,
        str(error),
    )


class _TruncatingModel(BaseChatModel):
    attempts: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-truncating"

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        self.attempts += 1
        message = AIMessage(content=[], response_metadata=_RESPONSES_INCOMPLETE)
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_empty_truncated_reply_reaches_tui_as_error() -> None:
    """The run fails, is not retried, saves nothing, and the TUI shows why."""
    model = _TruncatingModel()
    checkpointer = InMemorySaver()
    # Same relative order as `create_cli_agent`: the guard wraps the retry.
    graph = create_agent(
        model,
        middleware=[
            EmptyTruncatedReplyMiddleware(),
            CodeModelRetryMiddleware(max_retries=5),
        ],
        checkpointer=checkpointer,
    )
    config: Any = {"configurable": {"thread_id": _THREAD_ID}}
    frames: list[bytes] = []

    async def respond(_request: httpx.Request) -> httpx.Response:
        try:
            async for namespace, mode, data in graph.astream(
                {"messages": [HumanMessage(content="hi")]},
                config,
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
        except ModelOutputTruncatedError as exc:
            frames.append(b"event: error\ndata: " + json_dumpb(exc) + b"\n\n")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(frames),
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
                config=config,
                stream_mode=["messages", "updates", "custom"],
                subgraphs=True,
            ):
                pass

    assert model.attempts == 1
    state = await graph.aget_state(config)
    assert [type(m).__name__ for m in state.values["messages"]] == ["HumanMessage"]

    expected = (
        "ModelOutputTruncatedError: Model 'gpt-6.1-sol' stopped before replying: "
        "it reached its output token limit (max_output_tokens)."
    )
    assert format_agent_exception(caught.value).startswith(expected)
    app = DeepAgentsApp(agent=MagicMock())
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.tui.textual_adapter.execute_task_textual",
            new_callable=AsyncMock,
            side_effect=caught.value,
        ):
            await app._run_agent_task("hi")
            await pilot.pause()
        errors = app.query(ErrorMessage)
        assert any(expected in str(widget._content) for widget in errors)


def _glm_request() -> ModelRequest:
    model = MagicMock(spec=BaseChatModel)
    model.model_name = "accounts/fireworks/models/glm-5p2"
    model._get_ls_params.return_value = {"ls_provider": "fireworks"}
    return ModelRequest(
        model=cast("BaseChatModel", model),
        messages=[HumanMessage(content="run")],
        tools=[],
        system_prompt="base prompt",
        state={"messages": []},
        runtime=cast("Any", SimpleNamespace(context={"model": None})),
    )


def _glm_stack(
    responses: list[ModelResponse[Any]],
) -> tuple[ModelResponse[Any], int]:
    """Run the guard outside GLM recovery, as `create_cli_agent` orders them."""
    guard = EmptyTruncatedReplyMiddleware()
    recovery = _GlmTerminalStallRecovery()
    remaining = list(responses)

    def model_call(_request: ModelRequest) -> ModelResponse[Any]:
        return remaining.pop(0)

    response = guard.wrap_model_call(
        _glm_request(), lambda request: recovery.wrap_model_call(request, model_call)
    )
    return response, len(responses) - len(remaining)


def test_glm_recovery_runs_before_the_guard() -> None:
    stalled = _response(metadata={"finish_reason": "length"})
    recovered = _response(metadata={"finish_reason": "tool_calls"}, with_tool_call=True)

    response, calls = _glm_stack([stalled, recovered])

    assert calls == 2
    assert response is recovered


def test_guard_raises_when_glm_recovery_still_stalls() -> None:
    stalled = _response(metadata={"finish_reason": "length"})

    with pytest.raises(ModelOutputTruncatedError, match=r"\(length\)"):
        _glm_stack([stalled, stalled])
